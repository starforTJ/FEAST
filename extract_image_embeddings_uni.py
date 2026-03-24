import argparse
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import glob
import json
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

Image.MAX_IMAGE_PIXELS = None


def load_uni2_tile_encoder():
    """Load UNI2-h tile encoder for extracting spot embeddings."""
    print("Loading UNI2-h tile encoder...")

    try:
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        
        print("Loading UNI2-h model...")
        tile_encoder = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        tile_encoder = tile_encoder.to("cuda")
        tile_encoder.eval()
        print("UNI2-h tile encoder loaded successfully.")
        return tile_encoder
    except Exception as e:
        print(f"Error loading UNI2-h tile encoder: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_uni_transform(model):
    """Get transform for UNI2-h model."""
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return transform


def get_slide_list(data_dir: str) -> list[str]:
    """Get slide list from data directory."""
    wsi_dir = os.path.join(data_dir, "wsi")
    if not os.path.exists(wsi_dir):
        print(f"WSI directory not found: {wsi_dir}")
        return []
    
    slide_files = glob.glob(os.path.join(wsi_dir, "*.jpg"))
    slide_names = [os.path.splitext(os.path.basename(f))[0] for f in slide_files]
    
    print(f"Found {len(slide_names)} slides in {data_dir}")
    return sorted(slide_names)


def load_tissue_positions(data_dir: str, slide_name: str) -> pd.DataFrame | None:
    """Load tissue positions file (including pseudo_spots)."""
    tissue_file = os.path.join(data_dir, "pseudo_spots_linear", f"{slide_name}.csv")

    if not os.path.exists(tissue_file):
        print(f"Tissue positions file not found: {tissue_file}")
        return None
    
    try:
        tissue_df = pd.read_csv(tissue_file, header=0)

        if '0' in tissue_df.columns:
            tissue_df = tissue_df.rename(columns={'0': 'barcode'})
        
        if 'spot_name' in tissue_df.columns:
            tissue_df = tissue_df.rename(columns={'spot_name': 'barcode'})
        
        required_cols = ['in_tissue', 'pxl_col_in_fullres', 'pxl_row_in_fullres', 'is_pseudo']
        missing_cols = [col for col in required_cols if col not in tissue_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        tissue_spots = tissue_df[tissue_df['in_tissue'] == 1].copy()

        if 'is_pseudo' in tissue_spots.columns:
            num_original = (tissue_spots['is_pseudo'] == 0).sum()
            num_pseudo = (tissue_spots['is_pseudo'] == 1).sum()
            print(f"  Loaded {len(tissue_spots)} tissue spots for {slide_name} (original: {num_original}, pseudo: {num_pseudo})")
        else:
            print(f"  Loaded {len(tissue_spots)} tissue spots for {slide_name}")
        
        return tissue_spots

    except Exception as e:
        print(f"Error loading tissue positions for {slide_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_patches_from_slide(
    slide_path: str, tissue_positions: pd.DataFrame
) -> tuple[list[Image.Image], list[dict]]:
    """Extract patches from slide based on tissue spot positions."""
    try:
        wsi = np.array(Image.open(slide_path))
        wsi_height, wsi_width = wsi.shape[:2]
        
        print(f"  WSI size: {wsi_width} x {wsi_height}")
        
        patches = []
        valid_positions = []
        
        for idx, row in tissue_positions.iterrows():
            pxl_row = int(row['pxl_row_in_fullres'])
            pxl_col = int(row['pxl_col_in_fullres'])
            x, y = round(pxl_col), round(pxl_row)
            if x < 128:
                x = 128
            if y < 128:
                y = 128
            if x > wsi_width - 128:
                x = wsi_width - 128
            if y > wsi_height - 128:
                y = wsi_height - 128
            
            try:
                patch = wsi[y-128:y+128, x-128:x+128, :3]
                if patch.shape != (256, 256, 3):
                    print(f"Skipping patch with wrong shape: {patch.shape} at ({x}, {y})")
                    continue
                patch_image = Image.fromarray(patch.astype(np.uint8))
                
                patches.append(patch_image)
                valid_positions.append({
                    'barcode': row['barcode'],
                    'array_row': row['array_row'],
                    'array_col': row['array_col'],
                    'pxl_row': pxl_row,
                    'pxl_col': pxl_col,
                    'patch_center_x': x,
                    'patch_center_y': y
                })
                
            except Exception as e:
                print(f"Error extracting patch at ({x}, {y}): {e}")
                continue
        
        print(f"  Extracted {len(patches)} valid patches from {len(tissue_positions)} tissue spots")
        return patches, valid_positions
        
    except Exception as e:
        print(f"Error extracting patches from {slide_path}: {e}")
        return [], []


def preprocess_patches(patches: list[Image.Image], transform, batch_size: int = 1024) -> list[torch.Tensor]:
    """Preprocess patches and convert to batches (UNI2-h transform)."""
    patch_tensors = []
    for i, patch in enumerate(patches):
        try:
            if patch is None or patch.size == (0, 0):
                print(f"Skipping invalid patch {i}: {patch}")
                continue
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            
            patch_tensor = transform(patch)
            patch_tensors.append(patch_tensor)
            
        except Exception as e:
            print(f"Error processing patch {i}: {e}")
            print(f"   Patch info: size={patch.size if patch else 'None'}, mode={patch.mode if patch else 'None'}")
            continue
    if patch_tensors:
        all_tensors = torch.stack(patch_tensors)
        batches = []
        for i in range(0, len(all_tensors), batch_size):
            batch = all_tensors[i:i+batch_size]
            batches.append(batch)
        
        return batches
    else:
        return []


def extract_features_with_uni(tile_encoder: torch.nn.Module, patch_batches: list[torch.Tensor]) -> torch.Tensor:
    """Extract embeddings for each spot using UNI2-h tile encoder."""
    all_features = []

    with torch.no_grad():
        for batch in tqdm(patch_batches, desc="Extracting tile embeddings"):
            batch = batch.to("cuda")
            batch_embeddings = tile_encoder(batch)
            all_features.append(batch_embeddings.cpu())
    
    if all_features:
        return torch.cat(all_features, dim=0)  # (num_patches, 1536)
    else:
        return torch.empty(0, 1536)


def process_single_slide(
    tile_encoder: torch.nn.Module,
    transform,
    data_dir: str,
    slide_name: str,
    output_dir: str,
    patch_size: int = 224,
    batch_size: int = 1024
) -> bool:
    """Process a single slide."""
    print(f"\n=== Processing slide: {slide_name} ===")

    slide_path = os.path.join(data_dir, "wsi", f"{slide_name}.jpg")
    if not os.path.exists(slide_path):
        print(f"Slide file not found: {slide_path}")
        return False

    tissue_positions = load_tissue_positions(data_dir, slide_name)
    if tissue_positions is None or len(tissue_positions) == 0:
        print(f"No tissue positions found for {slide_name}")
        return False

    try:
        patches, valid_positions = extract_patches_from_slide(
            slide_path, tissue_positions
        )
        print(f"  Successfully extracted {len(patches)} patches and {len(valid_positions)} positions")
    except Exception as e:
        print(f"Error in extract_patches_from_slide: {e}")
        import traceback
        traceback.print_exc()
        return False

    if len(patches) == 0:
        print(f"No valid patches extracted for {slide_name}")
        return False

    try:
        patch_batches = preprocess_patches(patches, transform, batch_size=batch_size)
        print(f"  Successfully preprocessed patches into {len(patch_batches)} batches")
    except Exception as e:
        print(f"Error in preprocess_patches: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        features = extract_features_with_uni(tile_encoder, patch_batches)
        print(f"  Successfully extracted features: {features.shape}")
    except Exception as e:
        print(f"Error in extract_features_with_uni: {e}")
        import traceback
        traceback.print_exc()
        return False

    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)

    features_file = os.path.join(slide_output_dir, "uni_features.npy")
    np.save(features_file, features.numpy())

    positions_file = os.path.join(slide_output_dir, "tissue_positions.json")
    with open(positions_file, 'w') as f:
        json.dump(valid_positions, f, indent=2)

    num_original_spots = 0
    num_pseudo_spots = 0
    if 'is_pseudo' in tissue_positions.columns:
        num_original_spots = int((tissue_positions['is_pseudo'] == 0).sum())
        num_pseudo_spots = int((tissue_positions['is_pseudo'] == 1).sum())
    
    metadata = {
        'slide_name': slide_name,
        'num_patches': len(patches),
        'num_original_spots': num_original_spots,
        'num_pseudo_spots': num_pseudo_spots,
        'feature_dim': features.shape[1],
        'patch_size': patch_size,
        'batch_size': batch_size,
        'model': 'UNI2-h'
    }
    
    metadata_file = os.path.join(slide_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved features for {slide_name}: {features.shape}")
    print(f"   Features: {features_file}")
    print(f"   Positions: {positions_file}")
    print(f"   Metadata: {metadata_file}")
    
    return True


def process_dataset(
    dataset_name: str,
    data_root: str,
    output_root: str,
    patch_size: int = 224,
    batch_size: int = 1024
) -> None:
    """Process entire dataset."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    data_dir = os.path.join(data_root, dataset_name)
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    tile_encoder = load_uni2_tile_encoder()
    if tile_encoder is None:
        print("Failed to load UNI2-h tile encoder")
        return

    transform = get_uni_transform(tile_encoder)
    slide_list = get_slide_list(data_dir)
    if not slide_list:
        print(f"No slides found in {data_dir}")
        return

    print(f"Found {len(slide_list)} slides to process")
    successful_slides = 0
    failed_slides = []
    
    for slide_name in tqdm(slide_list, desc=f"Processing {dataset_name} slides"):
        try:
            success = process_single_slide(
                tile_encoder,
                transform,
                data_dir,
                slide_name,
                output_dir,
                patch_size=patch_size,
                batch_size=batch_size,
            )
            
            if success:
                successful_slides += 1
            else:
                failed_slides.append(slide_name)

        except Exception as e:
            print(f"Error processing {slide_name}: {e}")
            failed_slides.append(slide_name)
    
    print(f"\n{'='*60}")
    print(f"Dataset {dataset_name} processing completed!")
    print(f"Successful slides: {successful_slides}/{len(slide_list)}")

    if failed_slides:
        print(f"Failed slides: {len(failed_slides)}")
        for slide in failed_slides:
            print(f"   - {slide}")

    summary = {
        'dataset': dataset_name,
        'total_slides': len(slide_list),
        'successful_slides': successful_slides,
        'failed_slides': len(failed_slides),
        'failed_slide_names': failed_slides,
        'patch_size': patch_size,
        'batch_size': batch_size,
        'feature_dim': 1536,
        'model': 'UNI2-h'
    }
    
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="UNI2-h Feature Extraction")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory containing dataset folders")
    parser.add_argument("--output_root", type=str, default="./uni_feature", help="Output root directory")
    parser.add_argument("--datasets", type=str, nargs="+", default=["skin", "her2st", "stnet"], help="Dataset names to process")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    print("UNI2-h Feature Extraction")
    print("=" * 60)

    patch_size = args.patch_size
    batch_size = args.batch_size

    for dataset in args.datasets:
        try:
            process_dataset(
                dataset,
                data_root=args.data_root,
                output_root=args.output_root,
                patch_size=patch_size,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            continue

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()

