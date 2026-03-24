from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm
import sys


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def compute_metrics(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics (MSE, MAE, PCC) for gene expression prediction.

    Args:
        predictions: List of prediction tensors, each of shape (M, G) where M is masked spots, G is genes
        targets: List of target tensors, each of shape (M, G) where M is masked spots, G is genes

        Returns:
        Dictionary containing computed metrics:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - avg_gene_correlation: Average Pearson correlation coefficient across genes
    """
    if not predictions:
        return {
            'mse': 0.0,
            'mae': 0.0,
            'avg_gene_correlation': 0.0
        }

    # concatenate all predictions and targets
    preds = torch.cat(predictions, dim=0)  # (Total_masked_spots, G)
    gts = torch.cat(targets, dim=0)       # (Total_masked_spots, G)

    # basic metrics
    mse = F.mse_loss(preds, gts).item()
    mae = F.l1_loss(preds, gts).item()

    # convert to numpy for correlation calculation
    preds_np = preds.cpu().numpy()
    gts_np = gts.cpu().numpy()

    # gene-wise correlation (correlation for each gene across all spots)
    num_samples, num_genes = preds_np.shape

    # vectorized NaN/inf handling
    preds_clean = np.nan_to_num(preds_np, nan=0.0, posinf=0.0, neginf=0.0)
    gts_clean = np.nan_to_num(gts_np, nan=0.0, posinf=0.0, neginf=0.0)

    # center the data for all genes at once
    # preds_clean: (N, G), preds_mean: (1, G) -> preds_centered: (N, G)
    preds_mean = preds_clean.mean(axis=0, keepdims=True)  # (1, G)
    gts_mean = gts_clean.mean(axis=0, keepdims=True)      # (1, G)

    preds_centered = preds_clean - preds_mean  # (N, G)
    gts_centered = gts_clean - gts_mean        # (N, G)

    # compute covariance and variances for all genes simultaneously
    covariance = np.sum(preds_centered * gts_centered, axis=0)  # (G,)
    pred_var = np.sum(preds_centered ** 2, axis=0)              # (G,)
    gt_var = np.sum(gts_centered ** 2, axis=0)                  # (G,)

    # compute correlations: handle division by zero
    denominator = np.sqrt(pred_var * gt_var)
    correlations = np.divide(
        covariance, denominator,
        out=np.zeros_like(covariance),
        where=denominator != 0.0
    )

    # average correlation across all genes
    avg_corr = float(np.mean(correlations))

    return {
        'mse': mse,
        'mae': mae,
        'avg_correlation': avg_corr
    }


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    output_dir: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Train model with loss and metrics computed only on original spots (is_pseudo=0).
    Model inference uses all spots (is_pseudo=0 and 1) for attention computation,
    but model output and loss/metrics are computed only on original spots (is_pseudo=0).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloaders['train'], dataloaders['val'] = accelerator.prepare(model, optimizer, dataloaders['train'], dataloaders['val'])
    
    tqdm_disable = not accelerator.is_main_process or not sys.stdout.isatty()

    best_val_loss = float('inf')
    best_metrics: Dict[str, float] = {}
    best_epoch = -1

    history: List[Dict[str, float]] = []

    epoch_bar = tqdm(range(num_epochs), desc='Training', unit='epoch', disable=tqdm_disable)
    for epoch in epoch_bar:
        model.train()
        train_loss_sum = 0.0
        train_spots = 0

        train_bar = tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False, unit='batch', disable=tqdm_disable)
        
        for _, batch in enumerate(train_bar):
            # Use accelerate's accumulate context manager
            with accelerator.accumulate(model):
                features = batch['features']
                targets = batch['gt_expressions']
                barcodes = batch.get('barcodes')
                is_pseudo = batch.get('is_pseudo')

                # Forward pass: model output already returns only original spots (is_pseudo=0)
                predictions = model(features, barcodes=barcodes, is_pseudo=is_pseudo)

                # Loss calculation: use only data for original spots (is_pseudo=0)
                if is_pseudo is not None:
                    original_mask = (is_pseudo == 0)
                    if original_mask.sum() > 0:
                        # Filter targets to only original spots
                        targets_original = targets[original_mask]
                        loss = criterion(predictions, targets_original)
                        num_original_spots = original_mask.sum().item()
                    else:
                        # Set loss to 0 if there are no original spots
                        loss = torch.tensor(0.0, device=accelerator.device)
                        num_original_spots = 0
                else:
                    # Use all spots if is_pseudo is None
                    loss = criterion(predictions, targets)
                    num_original_spots = features.size(0)

                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping is performed only at actual sync time
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                
                # For logging: count only original spots
                if num_original_spots > 0:
                    train_loss_sum += loss.item() * num_original_spots
                    train_spots += num_original_spots
                
                # Update progress bar
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = train_loss_sum / max(train_spots, 1)

        model.eval()
        val_loss_sum = 0.0
        val_spots = 0  # Count only original spots (is_pseudo=0)
        val_predictions: List[torch.Tensor] = []
        val_targets: List[torch.Tensor] = []

        val_bar = tqdm(dataloaders['val'], desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False, unit='batch', disable=tqdm_disable)
        with torch.no_grad():
            for batch in val_bar:
                features = batch['features']
                targets = batch['gt_expressions']
                barcodes = batch.get('barcodes')
                is_pseudo = batch.get('is_pseudo')

                # Forward pass: model output already returns only original spots (is_pseudo=0)
                predictions = model(features, barcodes=barcodes, is_pseudo=is_pseudo)

                # Loss calculation: use only data for original spots (is_pseudo=0)
                if is_pseudo is not None:
                    original_mask = (is_pseudo == 0)
                    if original_mask.sum() > 0:
                        # Filter targets to only original spots (predictions already contain only original spots)
                        targets_original = targets[original_mask]
                        val_loss = criterion(predictions, targets_original)
                        num_original_spots = original_mask.sum().item()
                        val_loss_sum += val_loss.item() * num_original_spots
                        val_spots += num_original_spots
                    else:
                        val_loss = torch.tensor(0.0, device=accelerator.device)
                        num_original_spots = 0
                else:
                    # Use all spots if is_pseudo is None
                    val_loss = criterion(predictions, targets)
                    num_original_spots = features.size(0)
                    val_loss_sum += val_loss.item() * num_original_spots
                    val_spots += num_original_spots

                # Save predictions and targets for metrics calculation (only original spots)
                if is_pseudo is not None and original_mask.sum() > 0:
                    val_predictions.append(predictions.detach().cpu())
                    val_targets.append(targets[original_mask].detach().cpu())
                elif is_pseudo is None:
                    val_predictions.append(predictions.detach().cpu())
                    val_targets.append(targets.detach().cpu())
                
                # Update progress bar
                val_bar.set_postfix({'loss': f'{val_loss.item():.4f}'})

        val_loss = val_loss_sum / max(val_spots, 1)

        if scheduler is not None:
            scheduler.step()

        # Compute metrics: val_predictions and val_targets already contain only original spots
        metrics = compute_metrics(val_predictions, val_targets)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, **metrics})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            best_epoch = epoch

            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
            # Save best model metrics (mse, mae, corr) as JSON
            save_json(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'avg_correlation': metrics['avg_correlation'],              # Gene-wise correlation
                },
                output_dir / 'best_model_metrics.json',
            )

        # Update epoch progress bar
        epoch_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'corr': f'{metrics.get("avg_correlation", 0.0):.4f}',
        })

        # Save last epoch metrics for reference
        save_json(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics,
            },
            output_dir / 'last_epoch_metrics.json',
        )

    torch.save(model.state_dict(), output_dir / 'final_model.pt')

    save_json({'history': history, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'best_metrics': best_metrics}, output_dir / 'training_summary.json')

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        **best_metrics,
    }