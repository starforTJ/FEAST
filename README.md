<p align="center">
  <h1 align="center">FEAST: Fully Connected Expressive Attention for <br>Spatial Transcriptomics</h1>
  <h3 align="center"><b>CVPR 2026</b></h3>
  <p align="center">
    <h3 align="center">
      <a href="https://starfortj.github.io//"><strong>Taejin Jeong<sup>*</sup></strong></a> · 
      <a href="https://kyyle2114.github.io/"><strong>Joohyeok Kim<sup>*</sup></strong></a> · 
      <a href="https://rubato-yeong.github.io/"><strong>Jinyeong Kim</strong></a> · 
      <a href="https://kochanha.github.io/"><strong>Chanyoung Kim</strong></a> · 
      <a href="https://micv.yonsei.ac.kr/seongjae"><strong>Seong Jae Hwang</strong></a>
    </h3>
    <h3 align="center">
      Yonsei University
    </h3>
  </p>
  <br>
</p>
 
<be>

## Overview

This repository contains the official implementation of **FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics**.

**Note:** The overall experimental protocol is based on the code from **[MERGE](https://github.com/ags3927/MERGE) (Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from Whole Slide Histopathology Images)**, which was published at CVPR 2025. We gratefully acknowledge the authors for making their code publicly available.

## Stage 0: Environment Setup

Create and activate the **feast** conda environment (Python 3.10), then install dependencies from `requirements.txt`:

```bash
conda create -n feast python=3.10 -y
conda activate feast
pip install -r requirements.txt
```

## Stage 1: Data Preparation

We use three public spatial transcriptomics (ST) datasets: Her2ST, SKIN (SCC), and ST-Net.

The data file (`data.tar.gz`) can be downloaded from the **[MERGE](https://github.com/ags3927/MERGE) repository**. Extract it using:

```bash
tar -xvf data.tar.gz
```

After extraction, the directory structure should be organized as follows:

```bash
data
├── DATASET_NAME
│   ├── barcodes
│   ├── counts_8n
│   ├── ...
│   └── wsi
├── configs
├── engine
└── ... (other directories)
```

## Stage 2: Generate Pseudo-Spots and Expressions

We generate off-grid pseudo-spots between original spots and interpolate their gene expressions using linear interpolation. This process increases the spatial resolution of the data and facilitates improved model training.

Execute the following script to generate pseudo-spots and their expressions for all datasets:

```bash
python sample_off_grid_pseudo_spots.py
```

The script performs the following operations:
- Generates pseudo-spots by creating a fine grid between original spots
- Filters pseudo-spots based on distance to original spots
- Interpolates gene expressions for pseudo-spots using linear interpolation
- Saves combined spots (original + pseudo) to `data/{dataset}/pseudo_spots_linear/{slide_name}.csv`
- Saves combined expressions to `data/{dataset}/pseudo_counts_spcs_to_8n_linear/{slide_name}.npy`

**Important Note:** The gene expressions for pseudo-spots are generated using linear interpolation solely to match the number of spots with the extracted image embeddings. These interpolated labels are **not used during training**—only the original spot labels are used for loss computation. Pseudo-spots are used only for attention computation to provide richer spatial context.

The script processes all three datasets (`her2st`, `skin`, `stnet`) automatically.

## Stage 3: Extract UNI Image Embeddings

In this stage, we extract image embeddings using the pre-trained **UNI2-h** model (MahmoodLab/UNI2-h) from histopathology patches.

**Prerequisite:** UNI2-h is a gated model on Hugging Face. You must request and obtain access permission from the [UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h) before running the extraction script. Log in to Hugging Face (`huggingface-cli login`) after your access is granted.

Execute the following script to extract image embeddings:

```bash
python extract_image_embeddings_uni.py
```

**Arguments:**
- `--data_root`: Root directory containing dataset folders (default: `data`)
- `--output_root`: Output root directory (default: `./uni_feature`)
- `--datasets`: Dataset names to process (default: `skin her2st stnet`)
- `--patch_size`: Patch size (default: 224)
- `--batch_size`: Batch size for inference (default: 256)

**Example with custom paths:**

```bash
python extract_image_embeddings_uni.py \
    --data_root data \
    --output_root ./uni_feature \
    --datasets skin her2st stnet
```

The script performs the following operations:
- Loads the pre-trained UNI2-h tile encoder (no training required)
- Extracts 256×256 patches from WSI images for each spot (original + pseudo)
- Generates 1536-dimensional embeddings using UNI2-h
- Saves embeddings as `.npy` files for each slide

**Output structure:**

```bash
uni_feature/
├── skin/
│   ├── SLIDE_NAME/
│   │   ├── uni_features.npy       # (num_patches, 1536) embeddings
│   │   ├── tissue_positions.json  # barcode, coordinates, etc.
│   │   └── metadata.json          # slide info, feature_dim, etc.
│   └── processing_summary.json
├── her2st/
│   └── ...
└── stnet/
    └── ...
```

**Prerequisites:** Ensure Stage 2 (pseudo-spot generation) has been completed. The script expects tissue position files at `data/{dataset}/pseudo_spots_linear/{slide_name}.csv`.

## Stage 4: Training FEAST

In this final stage, we load the image embeddings and train the FEAST model. The training procedure performs 8-fold cross-validation for each dataset.

### Step 4.1: Prepare Configuration Files

Configuration file templates are located in the `configs/` directory. For each dataset, create or modify the configuration file. When using UNI embeddings, set `input_dim` to `1536` (UNI2-h embedding dimension).

**Example configuration for `her2st` dataset (`configs/config_her2st.yaml`):**

```yaml
Accelerate:
  gradient_accumulation_steps: 0  # 0 means accumulate over entire epoch

Data:
  dataset_name: her2st
  fold: all
  folds: 8
  num_genes: 250
  path: data/her2st
  slides: configs/slides_her2st.csv

General:
  output_dir_base: outputs
  seed: 3927

Model:
  dropout: 0.3
  beta: 1.5
  tau_neg: 0.6
  k_neighbors: 32
  input_dim: 1536  # UNI2-h embedding dimension (use 256 for ResNet)
  num_heads: 8
  num_layers: 3

Training:
  epochs: 1500
  loss_type: mse
  optimizer:
    lr: 1e-4
    weight_decay: 1e-5
  scheduler:
    T_max: 100
    eta_min: 1e-6
    type: cosine_annealing
```

### Step 4.2: Train FEAST Model

To train the FEAST model, execute the provided shell script from the project root directory:

```bash
bash scripts/her2st.sh
```

**Note:** The shell scripts automatically configure the device and config file. If `Data.fold` is set to `'all'` in the config file, the script trains all 8 folds sequentially.

### Step 4.3: Training Output Structure

```bash
outputs/
└── her2st/
    └── feast_her2st_tau0.6_beta1.5_k32/
        ├── fold0/
        │   ├── best_model.pt
        │   ├── best_model_metrics.json
        │   ├── final_model.pt
        │   ├── last_epoch_metrics.json
        │   └── training_summary.json
        ├── fold1/
        └── ...
```

### Step 4.4: Training All Datasets

```bash
bash scripts/her2st.sh
bash scripts/skin.sh
bash scripts/stnet.sh
```

**Note:** Ensure that Stage 3 (UNI image embedding extraction) has been completed. The training script expects embeddings in the configured embedding directory. You may need to update `utils/data_loader.py` to point to the UNI feature path (`uni_feature/{dataset_name}/`) instead of the ResNet path if using the default data loader.

### Important Notes

1. **Metrics Computation**: All metrics (MSE, MAE, correlation) are computed only on original spots (not pseudo-spots). However, the model uses both original and pseudo-spots for attention computation during inference.

2. **Model Selection**: The `best_model.pt` is saved based on validation loss. This model should be used for evaluation rather than `final_model.pt`.

3. **Training Time**: Training may take several hours depending on the dataset size and hardware. Each fold is trained independently.
