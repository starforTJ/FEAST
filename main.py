from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from engine.trainer import train_model
from model.feast import FEAST
from utils.misc import load_config, set_seed
from utils.data_loader import create_dataloaders, dataset_slide_counts, split_slides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FEAST Model")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/configs/skin_transform.yaml',
        help='Path to YAML configuration file.',
    )
    return parser.parse_args()


def build_model(config: Dict, device: torch.device) -> nn.Module:
    model_cfg = config['Model']

    model = FEAST(
        input_dim=int(model_cfg['input_dim']),
        num_blocks=int(model_cfg['num_layers']),
        num_heads=int(model_cfg['num_heads']),
        dropout=float(model_cfg['dropout']),
        num_genes=int(config['Data']['num_genes']),
        tau_neg=float(model_cfg['tau_neg']),
        beta=float(model_cfg['beta']),
        k_neighbors=int(model_cfg['k_neighbors']),
    )

    return model.to(device)


def build_criterion(config: Dict) -> nn.Module:
    loss_type = config['Training'].get('loss_type', 'mse').lower()
    if loss_type == 'mse':
        return nn.MSELoss()
    if loss_type == 'mae':
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def build_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    opt_cfg = config['Training'].get('optimizer', {})
    lr = float(opt_cfg.get('lr', 1e-4))
    weight_decay = float(opt_cfg.get('weight_decay', 1e-5))
    
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler | None:
    scheduler_cfg = config['Training'].get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'cosine_annealing').lower()

    if scheduler_type in {'none', 'off'}:
        return None
    if scheduler_type in {'cosine', 'cosine_annealing'}:
        t_max = int(scheduler_cfg.get('T_max', 50))
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def folds_to_process(config: Dict) -> List[int]:
    fold_value = config['Data']['fold']
    if isinstance(fold_value, str) and fold_value.lower() == 'all':
        return list(range(int(config['Data']['folds'])))
    
    if isinstance(fold_value, str) and ',' in fold_value:
        return [int(fold) for fold in fold_value.split(',')]
    
    return [int(fold_value)]


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_cfg = config['Model']
    dataset_name = config['Data']['dataset_name']
    output_dir_name = f"feast_{dataset_name}_tau{model_cfg['tau_neg']}_beta{model_cfg['beta']}_k{model_cfg['k_neighbors']}"
    output_root = Path(config['General'].get('output_dir_base', 'outputs')) / dataset_name / output_dir_name
    
    folds = folds_to_process(config)

    for fold in folds:
        print(f"\n=== Fold {fold} ===")
        set_seed(int(config['General']['seed']))
        
        fold_output = output_root / f'fold{fold}'
        
        fold_output.mkdir(parents=True, exist_ok=True)
        print(f"Fold {fold} output directory created: {fold_output}")

        train_slides, val_slides = split_slides(config, fold)
        counts_info = dataset_slide_counts(train_slides, val_slides)
        
        print(f"Train slides: {counts_info['train']}, Val slides: {counts_info['val']}")
        print(f"Train slides: {train_slides} \nVal slides: {val_slides} \n")

        dataloaders = create_dataloaders(
            train_slides,
            val_slides,
            config,
            fold=fold,
            num_workers=4
        )

        model = build_model(config, device)
        criterion = build_criterion(config).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        gradient_accumulation_steps = config['Accelerate'].get('gradient_accumulation_steps', 1)
        
        # If gradient_accumulation_steps is 0 or negative, use epoch length (accumulate whole epoch)
        if gradient_accumulation_steps <= 0:
            gradient_accumulation_steps = len(dataloaders['train'])
            print(f"Gradient accumulation: Using epoch length ({gradient_accumulation_steps} batches = train slides)")
        else:
            print(f"Gradient accumulation steps: {gradient_accumulation_steps} (train slides: {len(dataloaders['train'])})")

        metrics = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=int(config['Training']['epochs']),
            output_dir=fold_output,
            scheduler=scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        print(f"Fold {fold} best val loss: {metrics['best_val_loss']:.6f}")
        print(f'-------------------------------- FOLD {fold} END -------------------------------- \n')