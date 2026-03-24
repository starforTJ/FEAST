import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator


def estimate_affine_transform(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Estimates the affine transformation from array coordinates to pixel coordinates.

    It uses all available points to compute a robust transformation matrix
    using the RANSAC algorithm, which provides better accuracy than using only
    three points.

    Args:
        df (pd.DataFrame): DataFrame with spot data, including array and pixel coordinates.

    Returns:
        np.ndarray: The 2x3 affine transformation matrix.

    Raises:
        RuntimeError: If the affine transformation cannot be estimated.
    """
    # use all available points to estimate the affine transformation for better accuracy
    src_pts = df[["array_col", "array_row"]].values.astype(np.float32)
    dst_pts = (
        df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float32)
    )

    # cv2.estimateAffine2D returns the transformation matrix and a mask of inliers
    # we use RANSAC for a more robust estimation against outliers.
    transform_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

    if transform_matrix is None:
        raise RuntimeError("Could not estimate affine transformation. Check input points.")

    return transform_matrix


def transform_coords(coords: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Applies an affine transformation to a set of coordinates.

    Args:
        coords (np.ndarray): An array of (col, row) coordinates.
        transform_matrix (np.ndarray): The 2x3 affine transformation matrix.

    Returns:
        np.ndarray: The transformed pixel coordinates.
    """
    # add a homogeneous coordinate (1) to each point
    homogeneous_coords = np.c_[coords, np.ones(coords.shape[0])]
    # apply the transformation
    return (transform_matrix @ homogeneous_coords.T).T


def generate_pseudo_spots(
    spots_path: Path,
    output_path: Path,
) -> None:
    """
    Generates pseudo-spots and saves them to a new CSV file.

    Args:
        spots_path (Path): Path to the tissue positions CSV file.
        output_path (Path): Path to save the output CSV with pseudo-spots.
    """
    # --- Load Data ---
    # robustly load spots csv by setting first column as index, then resetting it
    # this handles cases where the spot name column is named '0' or is unnamed.
    df_spots = pd.read_csv(spots_path, index_col=0)
    df_spots = df_spots.reset_index()
    df_spots = df_spots.rename(columns={df_spots.columns[0]: "spot_name"})

    # --- Estimate Coordinate Transformation ---
    affine_matrix = estimate_affine_transform(df_spots)

    # --- Generate Fine Grid for Pseudo-Spots ---
    print("Generating pseudo-spots...")
    array_cols = np.sort(df_spots["array_col"].unique())
    array_rows = np.sort(df_spots["array_row"].unique())

    # create midpoints
    mid_cols = (array_cols[:-1] + array_cols[1:]) / 2
    mid_rows = (array_rows[:-1] + array_rows[1:]) / 2

    # create fine grid coordinates
    fine_cols = np.sort(np.concatenate([array_cols, mid_cols]))
    fine_rows = np.sort(np.concatenate([array_rows, mid_rows]))

    # create fine meshgrid
    fine_mesh_col, fine_mesh_row = np.meshgrid(fine_cols, fine_rows)

    # --- Identify Pseudo-Spots ---
    # create a mask to identify original points
    is_original_point = np.isin(fine_mesh_col, array_cols) & np.isin(
        fine_mesh_row, array_rows
    )

    # get pseudo-spot coordinates
    pseudo_spot_cols = fine_mesh_col[~is_original_point].flatten()
    pseudo_spot_rows = fine_mesh_row[~is_original_point].flatten()
    pseudo_array_coords = np.vstack([pseudo_spot_cols, pseudo_spot_rows]).T

    # transform to pixel coordinates
    pseudo_pixel_coords = transform_coords(pseudo_array_coords, affine_matrix)

    # --- Calculate Average Distance Between Original Spots (in pixel coordinates) ---
    print("Calculating average distance between original spots in pixel coordinates...")
    original_pixel_coords = df_spots[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
    original_kdtree = KDTree(original_pixel_coords)
    
    # find nearest neighbor for each original spot (excluding itself)
    # query with k=2 to get the spot itself and its nearest neighbor
    distances, _ = original_kdtree.query(original_pixel_coords, k=2)
    # exclude the distance to itself (which is 0), take the nearest neighbor distance
    nearest_neighbor_distances = distances[:, 1]
    avg_distance = np.mean(nearest_neighbor_distances)
    print(f"Average distance to nearest neighbor (pixels): {avg_distance:.2f}")

    # --- Filter Pseudo-Spots Based on Distance to Original Spots ---
    print("Filtering pseudo-spots based on distance threshold...")
    pseudo_spots = []
    filtered_count = 0
    
    for i in tqdm(range(len(pseudo_array_coords)), dynamic_ncols=True):
        pxl_col, pxl_row = pseudo_pixel_coords[i]
        pseudo_pixel_coord = np.array([[pxl_col, pxl_row]])  # keep 2D shape for query
        distance_to_nearest, _ = original_kdtree.query(pseudo_pixel_coord, k=1)
        distance_to_nearest = distance_to_nearest[0]  # extract scalar
        
        # only include pseudo-spot if distance is less than or equal to average distance
        if distance_to_nearest <= avg_distance:
            pseudo_spots.append(
                {
                    "spot_name": f"{pseudo_spot_rows[i]}x{pseudo_spot_cols[i]}",
                    "in_tissue": 1,
                    "array_row": pseudo_spot_rows[i],
                    "array_col": pseudo_spot_cols[i],
                    "pxl_col_in_fullres": pxl_col,
                    "pxl_row_in_fullres": pxl_row,
                    "is_pseudo": 1,
                }
            )
        else:
            filtered_count += 1

    # --- Combine and Save Data ---
    print(f"Found {len(pseudo_spots)} pseudo-spots (filtered out {filtered_count} spots).")
    df_pseudo = pd.DataFrame(pseudo_spots)
    
    # add is_pseudo column to original spots
    df_spots["is_pseudo"] = 0
    
    # ensure column order is the same
    cols = [
        "spot_name", 
        "in_tissue", 
        "array_row", 
        "array_col", 
        "pxl_col_in_fullres", 
        "pxl_row_in_fullres", 
        "is_pseudo"
    ]
    df_spots = df_spots[cols]
    if not df_pseudo.empty:
        df_pseudo = df_pseudo[cols]
    
    # combine original and pseudo spots
    df_combined = pd.concat([df_spots, df_pseudo], ignore_index=True)

    # save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Saved combined spots to: {output_path}")


def calculate_pseudo_expressions(
    df_spots: pd.DataFrame,
    original_expressions: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calculates gene expressions for pseudo-spots using linear interpolation.

    Args:
        df_spots (pd.DataFrame): DataFrame containing both original and pseudo-spots.
        original_expressions (np.ndarray): NumPy array of gene expressions for original spots.

    Returns:
        Optional[np.ndarray]: A NumPy array with the calculated expressions for
        pseudo-spots, or None if no pseudo-spots are present.
    """
    # --- Separate Original and Pseudo Spots ---
    df_original = df_spots[df_spots["is_pseudo"] == 0].copy()
    df_pseudo = df_spots[df_spots["is_pseudo"] == 1].copy()

    if df_pseudo.empty:
        print("No pseudo-spots found to process.")
        return None

    # --- Validate Data Alignment ---
    if len(df_original) != original_expressions.shape[0]:
        raise ValueError(
            "Mismatch between the number of original spots "
            f"({len(df_original)}) and expression data rows "
            f"({original_expressions.shape[0]})."
        )

    # --- Get Coordinates ---
    original_coords = df_original[["array_row", "array_col"]].values
    pseudo_coords = df_pseudo[["array_row", "array_col"]].values

    # --- Linear Interpolation Using Delaunay Triangulation ---
    num_genes = original_expressions.shape[1]
    num_pseudo = len(pseudo_coords)
    pseudo_expressions = np.zeros((num_pseudo, num_genes), dtype=np.float32)

    for gene_idx in tqdm(range(num_genes), desc="Interpolating genes", leave=False, dynamic_ncols=True):
        interpolator = LinearNDInterpolator(
            original_coords,
            original_expressions[:, gene_idx],
            fill_value=np.nan,
        )
        pseudo_expressions[:, gene_idx] = interpolator(pseudo_coords)

    # handle points outside the convex hull (fill with nearest neighbor)
    nan_mask = np.isnan(pseudo_expressions).any(axis=1)
    if nan_mask.any():
        print(
            f"Warning: {nan_mask.sum()} pseudo-spots are outside the convex hull. "
            "Filling with nearest neighbor values."
        )
        kdtree = KDTree(original_coords)
        _, indices = kdtree.query(pseudo_coords[nan_mask], k=1)
        if indices.ndim > 1:
            indices = indices.flatten()
        pseudo_expressions[nan_mask] = original_expressions[indices]

    return pseudo_expressions


def generate_pseudo_expression_for_slide(
    spots_path: Path,
    counts_path: Path,
    output_path: Path,
) -> None:
    """
    Generates and saves pseudo-spot expressions for a single slide.

    Orchestrates the loading, calculation, combination, and saving of
    gene expression data using linear interpolation.

    Args:
        spots_path (Path): Path to the CSV file with combined spot data.
        counts_path (Path): Path to the .npy file with original expression data.
        output_path (Path): Path to save the final combined expression .npy file.
    """
    print(f"Processing slide: {spots_path.stem}")

    # --- Load Data ---
    df_spots = pd.read_csv(spots_path)
    original_expressions = np.load(counts_path)

    # --- Calculate Pseudo-Spot Expressions ---
    pseudo_expressions = calculate_pseudo_expressions(df_spots, original_expressions)

    # --- Combine Original and Pseudo Expressions ---
    num_total_spots = len(df_spots)
    num_genes = original_expressions.shape[1]

    # create an empty array to store the final combined results
    combined_expressions = np.zeros((num_total_spots, num_genes), dtype=np.float32)

    # get boolean masks to locate original and pseudo spots in the combined dataframe
    original_indices_mask = df_spots["is_pseudo"] == 0

    # fill the array, ensuring the order matches the CSV file
    combined_expressions[original_indices_mask] = original_expressions

    if pseudo_expressions is not None:
        pseudo_indices_mask = df_spots["is_pseudo"] == 1
        combined_expressions[pseudo_indices_mask] = pseudo_expressions

    # --- Final Validation ---
    if combined_expressions.shape[0] != len(df_spots):
        raise RuntimeError(
            "Final expression data rows do not match total spots count. "
            f"Expected {len(df_spots)}, Got {combined_expressions.shape[0]}."
        )

    # --- Save the Result ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, combined_expressions)
    print(f"Successfully saved combined expressions to: {output_path}")


if __name__ == "__main__":
    # --- Configuration ---
    interpolation_method = "linear"

    # generate pseudo spots and expressions for all datasets
    for dataset in ["her2st", "skin", "stnet"]:
        print(f"\n--- Generating Pseudo-Spots and Expressions for {dataset.upper()} ---")
        data_root = Path(f"./data/{dataset}")
        
        tissue_positions_dir = data_root / "tissue_positions"
        counts_dir = data_root / "counts_spcs_to_8n"
        
        if not tissue_positions_dir.is_dir():
            print(f"Directory not found: {tissue_positions_dir}. Skipping dataset.")
            continue
            
        slide_ids = sorted([f.stem for f in tissue_positions_dir.glob("*.csv")])
        
        if not slide_ids:
            print(f"No original spot CSV files found in {tissue_positions_dir}. Skipping.")
            continue

        for slide_id in tqdm(slide_ids, desc=f"Processing {dataset}", unit="slide", dynamic_ncols=True):
            print(f"\n--- Processing slide: {slide_id} ---")
            
            # --- Define paths for the entire pipeline ---
            spots_path = tissue_positions_dir / f"{slide_id}.csv"
            counts_path = counts_dir / f"{slide_id}.npy"
            
            pseudo_spots_output_path = data_root / f"pseudo_spots_{interpolation_method}" / f"{slide_id}.csv"
            pseudo_counts_output_path = data_root / f"pseudo_counts_spcs_to_8n_{interpolation_method}" / f"{slide_id}.npy"

            # --- Step 1: Generate pseudo spots ---
            try:
                if not spots_path.exists():
                    print(f"Skipping spot generation: Original spots file not found at {spots_path}")
                    continue

                print(f"Generating pseudo spots for {slide_id}...")
                generate_pseudo_spots(
                    spots_path=spots_path,
                    output_path=pseudo_spots_output_path,
                )
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error generating pseudo spots for {slide_id}: {e}")
                continue

            # --- Step 2: Generate pseudo expressions ---
            try:
                if not counts_path.exists():
                    print(f"Skipping expression generation: Original counts file not found at {counts_path}")
                    continue
                if not pseudo_spots_output_path.exists():
                    print(f"Skipping expression generation: Pseudo spots file not found at {pseudo_spots_output_path}")
                    continue

                print(f"Generating pseudo expressions for {slide_id}...")
                generate_pseudo_expression_for_slide(
                    spots_path=pseudo_spots_output_path,
                    counts_path=counts_path,
                    output_path=pseudo_counts_output_path,
                )
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                print(f"Error generating pseudo expressions for {slide_id}: {e}")
                continue
    
    print("\n--- All datasets processed successfully! ---")