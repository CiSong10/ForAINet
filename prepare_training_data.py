"""
This script takes post-QC (manual labeling) tree point cloud data, cleans the data, and merges into one ply file to be trained on.
"""

from pathlib import Path
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import logging


def read_clean_ply(file_path: Path):
    """Read a PLY file and return a standardized DataFrame with ['x', 'y', 'z', 'treeID', 'semantic_seg']"""
    ply = PlyData.read(file_path)
    data = ply['vertex'].data

    # check if field names are correct
    required_fields = {'x', 'y', 'z', 'semantic_pred', 'instance_pred'}
    fields = set(data.dtype.names)
    missing = required_fields - fields
    if missing:
        raise NameError(f"Invalid field names in Tree {str(file_path.name)}: fields = {fields}")
    
    # data['instance_pred'] = treeID

    # Output np.array with correct standardized field names
    df = pd.DataFrame({
        'x': data['x'],
        'y': data['y'],
        'z': data['z'],
        'treeID': data['instance_pred'],
        'semantic_seg': data['semantic_pred']
    })

    return df


def clean_point_cloud(df):
    """
    Remove duplicates and resolve conflicts.
    Rule: If multiple rows have same (x, y, z), keep the one with the highest treeID.
    """

    return df.sort_values('treeID', ascending=False).drop_duplicates(subset=['x', 'y', 'z'])


def get_intensity(df: pd.DataFrame, reference_data):
    """Assign intensity from reference_data to array by inner-joining on x,y,z"""

    df_ref = pd.DataFrame({
        'x': reference_data['x'],
        'y': reference_data['y'],
        'z': reference_data['z'],
        'intensity': reference_data['intensity']
    })

    df_merged = pd.merge(df, df_ref, on=['x', 'y', 'z'], how='inner', sort=False)
    
    num_dropped = len(df) - len(df_merged)
    if num_dropped > 0:
        logging.warning(f"Discarded {num_dropped} unmatched points (not found in raw data)")

    return df_merged


def clean_semantic_ids(array):
    """
    Cleans the semantic_seg field in a structured array:
    1. Shifts semantic classes from 0-4 to 1-5 if needed.
    2. Reassigns invalid values (-1, 0, or NaN) to class 1 (ground).
    """
    # check semantic classies
    semantic = array['semantic_seg'].astype(float)

    # Step 1: Shift labels if in 0–4 range
    semantic_values = np.unique(semantic[~np.isnan(semantic)])  # exclude nan for checking
    logging.info(f"Current semantic classes: {semantic_values}")
    if 5 not in semantic_values:
        logging.info("Shifts semantic classes from 0-4 to 1-5, by adding +1 to all semantic labels")
        semantic += 1

    # Step 2: Handle invalid values (NaN or 0)
    if np.isnan(semantic).any() or 0 in semantic:
        logging.warning("Found NaN or 0 in semantic_seg. Reassigning to 2 (ground)")
        semantic[np.isnan(semantic)] = 2
        semantic[semantic == 0] = 2

    # Final check: ensure semantic_seg contains only valid classes [1, 2, 3, 4, 5]
    valid_classes = {1.0, 2.0, 3.0, 4.0, 5.0}
    unique_classes = set(np.unique(semantic))
    if not unique_classes.issubset(valid_classes):
        raise ValueError(f"Invalid semantic classes found: {unique_classes - valid_classes}")

    # Assign back
    array['semantic_seg'] = semantic.astype(array['semantic_seg'].dtype)

    return array


def save_point_clouds(data, output_path):
    if isinstance(data, pd.DataFrame):
        data = data[['x', 'y', 'z', 'intensity', 'semantic_seg', 'treeID']].to_records(index=False)

    el = PlyElement.describe(data, 'vertex')
    PlyData([el], text=False).write(str(output_path))
    logging.info(f"Saved annotated point cloud to {output_path}")

def minimal_export(data, buffer_radius=2.0, tree_ids=None):
    """
    Export valid tree points and non-tree surroundings within a buffer.

    Parameters:
        data (pd.DataFrame or np.ndarray): Input points with columns
            ['x', 'y', 'z', 'intensity', 'semantic_seg', 'treeID']
        buffer_radius (float): Buffer radius around trees for non-tree points (meters)
        tree_ids (list, array, or None): Specific treeIDs to export. Default None = all trees (treeID > 0)
    
    Returns:
        pd.DataFrame: Filtered points
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Select tree points
    if tree_ids is None:
        tree_points = data[data['treeID'] > 0]
    else:
        tree_points = data[data['treeID'].isin(tree_ids)]
    
    # Bounding box of trees with buffer
    min_x = tree_points['x'].min() - buffer_radius
    max_x = tree_points['x'].max() + buffer_radius
    min_y = tree_points['y'].min() - buffer_radius
    max_y = tree_points['y'].max() + buffer_radius

    # Candidate non-tree points
    non_tree_points = data[(data['semantic_seg'].isin([1, 2])) & (data['treeID'] == 0)]

    # Keep only points inside bounding box
    nearby_non_tree_points = non_tree_points[
        (non_tree_points['x'] >= min_x) & (non_tree_points['x'] <= max_x) &
        (non_tree_points['y'] >= min_y) & (non_tree_points['y'] <= max_y)
    ]

    # Combine tree points and nearby non-tree points
    minimal_df = pd.concat([tree_points, nearby_non_tree_points], ignore_index=True)
    return minimal_df

def main():
    logging.basicConfig(level=logging.INFO)

    DATA_DIR = "data_QC/QC1"
    RAW_PLY_PATH = "data_QC/TranCanadaHwy.ply"
    OUTPUT_DIR = "data_QC/"
    output_path = Path(OUTPUT_DIR) / "TranCanadaHwy_annotated.ply"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Step 1: Read, clean and merge all PLYs
    merged_list = []
    for fname in sorted(Path(DATA_DIR).iterdir()):
        if fname.suffix == ".ply":
            df = read_clean_ply(fname)
            merged_list.append(df)

    df = pd.concat(merged_list, ignore_index=True)

    # Step 2: Deduplicate and resolve label conflicts
    df = clean_point_cloud(df)

    # Step 3: Add intensity using raw PLY
    raw_ply = PlyData.read(RAW_PLY_PATH)
    raw_data = raw_ply['vertex'].data
    df = get_intensity(df, raw_data)

    structured_array = df[['x', 'y', 'z', 'intensity', 'semantic_seg', 'treeID']].to_records(index=False)
    
    # Step 4: Adjust segmentation IDs.
    structured_array = clean_semantic_ids(structured_array)

    # Step 5: Save
    save_point_clouds(structured_array, output_path)

    # Step 6 (optional): Export valid trees and their surroundings (2m buffer) only
    minimal_df = minimal_export(structured_array)
    save_point_clouds(minimal_df, output_path= Path(OUTPUT_DIR) / "TranCanadaHwy_1_annotated_train.ply")



if __name__ == "__main__":
    main()
