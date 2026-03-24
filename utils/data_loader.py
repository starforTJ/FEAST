from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold

def split_slides(config: Dict, fold: int) -> Tuple[List[str], List[str]]:
    """Return train/validation slide names for the given fold."""
    slides_csv = config['Data']['slides']
    slides = pd.read_csv(slides_csv, header=None)[0].tolist()
    slides = sorted(slides)

    n_splits = int(config['Data']['folds'])
    seed = int(config['General']['seed'])

    slides_arr = np.asarray(slides)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = list(kf.split(slides_arr))[fold]
    train_slides = slides_arr[train_idx].tolist()
    val_slides = slides_arr[val_idx].tolist()
    
    print(f"KFold split: Train slides: {len(train_slides)}, Val slides: {len(val_slides)}")
    
    return train_slides, val_slides


# cached feature loader
class SlideDataset(Dataset):
    """
    Dataset that loads all spots from a single slide (read .npy files).
    In both train and val phases, all spots (original + pseudo) are used.
    """
    
    # class-level cache for CSV files (shared across all instances)
    _barcode_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    _pseudo_spots_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}

    def __init__(
        self,
        slides: Sequence[str],
        config: Dict,
        phase: str,
        fold: int,
    ) -> None:
        if phase not in {"train", "val"}:
            raise ValueError("phase must be 'train' or 'val'")

        self.slides = list(slides)
        self.config = config
        self.phase = phase
        self.fold = fold
        self.data_path = Path(config['Data']['path'])
        dataset_name = config['Data']['dataset_name']

        self.pseudo_data_path = Path("data") / dataset_name
        self.pseudo_feature_root = Path("uni_feature") / dataset_name

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, index: int) -> Dict[str, object]:
        slide = self.slides[index]

        features = self._load_features(slide)
        counts = self._load_counts(slide)
        barcodes = self._load_barcodes(slide)
        pseudo_spots_info = self._load_pseudo_spots_info(slide)

        if barcodes is not None and len(barcodes) != counts.shape[0]:
            raise ValueError(
                f"Spot count mismatch for {slide}: "
                f"barcodes={len(barcodes)}, counts={counts.shape[0]}. "
                f"The order of pseudo_counts_spcs_to_8n and pseudo_spots must match."
            )
        if len(features) != counts.shape[0]:
            raise ValueError(
                f"Feature/Count mismatch for {slide}: "
                f"features={len(features)}, counts={counts.shape[0]}. "
                f"The number of features and counts must match."
            )

        nonzero_mask = counts.sum(axis=1) > 0
        features = features[nonzero_mask]
        counts = counts[nonzero_mask]

        if barcodes is not None:
            barcodes = barcodes[nonzero_mask]

        # Extract is_pseudo information (needed in both train and val)
        is_pseudo_tensor = None
        if pseudo_spots_info is not None:
            # is_pseudo information before applying nonzero_mask
            original_is_pseudo = pseudo_spots_info['is_pseudo'].values
            # Apply nonzero_mask
            is_pseudo_tensor = torch.tensor(original_is_pseudo[nonzero_mask], dtype=torch.long)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        counts_tensor = torch.tensor(counts, dtype=torch.float32)

        barcodes_list: Optional[List[str]]
        if barcodes is None:
            barcodes_list = None
        else:
            barcodes_list = barcodes.astype(str).tolist()

        result = {
            'slide_name': slide,
            'features': features_tensor,
            'gt_expressions': counts_tensor,
            'barcodes': barcodes_list,
        }
        
        # Add is_pseudo information in both train and val
        if is_pseudo_tensor is not None:
            result['is_pseudo'] = is_pseudo_tensor

        return result

    def _load_features(self, slide: str) -> np.ndarray:
        """
        Load UNI features for all spots (original + pseudo) from pseudo_feature_root.
        """
        feature_file = self.pseudo_feature_root / slide / "uni_features.npy"
        if not feature_file.exists():
            raise FileNotFoundError(f"UNI feature file not found: {feature_file}")
        return np.load(feature_file)

    def _load_counts(self, slide: str) -> np.ndarray:
        """
        Load pseudo counts data.
        
        Note: The order of pseudo_counts_spcs_to_8n must match the order of pseudo_spots.
        The order has been confirmed to match, but matching logic may be needed if the order differs.
        """
        counts_file = self.pseudo_data_path / 'pseudo_counts_spcs_to_8n_linear' / f'{slide}.npy'
        if not counts_file.exists():
            raise FileNotFoundError(f"Pseudo counts file not found: {counts_file}")
        return np.load(counts_file)

    def _load_barcodes(self, slide: str) -> Optional[np.ndarray]:
        """
        Load spot_name from pseudo spots file with caching.
        
        Note:
        - The first column of pseudo_spots file is 'spot_name' (original tissue_positions has 0 as first column)
        - The returned order must match the order of pseudo_counts_spcs_to_8n
        - Returns all spots (real + pseudo)
        - Cached at class level for performance (key: (dataset_name, slide))
        """
        dataset_name = self.config['Data']['dataset_name']
        cache_key = (dataset_name, slide)
        
        # check cache first
        if cache_key in self._barcode_cache:
            return self._barcode_cache[cache_key]
        
        # load from file if not cached
        barcode_file = self.pseudo_data_path / 'pseudo_spots_linear' / f'{slide}.csv'
        if not barcode_file.exists():
            self._barcode_cache[cache_key] = None
            return None

        try:
            pseudo_spots_df = pd.read_csv(barcode_file, index_col='spot_name')
            # Return names of all spots (real + pseudo)
            # Order must match pseudo_counts_spcs_to_8n
            barcodes = pseudo_spots_df.index.to_numpy(dtype=str)
            self._barcode_cache[cache_key] = barcodes
            return barcodes
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read pseudo_spots file {barcode_file}: {exc}")

    def _load_pseudo_spots_info(self, slide: str) -> Optional[pd.DataFrame]:
        """
        Load full information from pseudo spots file (including is_pseudo column) with caching.
        
        Returns:
            pseudo_spots DataFrame (index='spot_name', columns include 'is_pseudo')
        
        Note:
            - Cached at class level for performance (key: (dataset_name, slide))
        """
        dataset_name = self.config['Data']['dataset_name']
        cache_key = (dataset_name, slide)
        
        # check cache first
        if cache_key in self._pseudo_spots_cache:
            return self._pseudo_spots_cache[cache_key]
        
        # load from file if not cached
        barcode_file = self.pseudo_data_path / 'pseudo_spots_linear' / f'{slide}.csv'
        if not barcode_file.exists():
            self._pseudo_spots_cache[cache_key] = None
            return None

        try:
            pseudo_spots_df = pd.read_csv(barcode_file, index_col='spot_name')
            self._pseudo_spots_cache[cache_key] = pseudo_spots_df
            return pseudo_spots_df
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read pseudo_spots file {barcode_file}: {exc}")
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear all cached data (barcodes and pseudo spots info).
        
        Useful for memory management in long-running processes like hyperparameter sweeps.
        """
        cls._barcode_cache.clear()
        cls._pseudo_spots_cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, int]:
        """
        Get information about current cache size.
        
        Returns:
            Dictionary with cache sizes for debugging/monitoring.
        """
        return {
            'barcode_cache_size': len(cls._barcode_cache),
            'pseudo_spots_cache_size': len(cls._pseudo_spots_cache),
        }


def create_dataloaders(
    train_slides: Sequence[str],
    val_slides: Sequence[str],
    config: Dict,
    fold: int,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    train_dataset = SlideDataset(
        train_slides,
        config,
        phase='train',
        fold=fold,
    )

    val_dataset = SlideDataset(
        val_slides,
        config,
        phase='val',
        fold=fold,
    )

    def _collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        # batch_size is 1, so return the single element for convenience
        return batch[0]

    return {
        'train': DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, collate_fn=_collate_fn),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_collate_fn),
    }


def dataset_slide_counts(train_slides: Sequence[str], val_slides: Sequence[str]) -> Dict[str, int]:
    return {
        'train': len(train_slides),
        'val': len(val_slides),
    }