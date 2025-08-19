"""
This script takes preliminary results predicted on pre-trained LiDAR point cloud data with segmation IDs assigned,
seperate point cloud by treeIDs, and outputs a spreadsheet of tree neighbors.
"""

import csv
import os
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from pathlib import Path
import distinctipy

def ply_to_xyz(data):
    return np.vstack((data['x'], data['y'], data['z'])).T


def check_xyz_alignment(data1, data2):
    xyz1 = ply_to_xyz(data1)
    xyz2 = ply_to_xyz(data2)

    if np.array_equal(xyz1, xyz2):
        print("✅ The xyz coordinates match exactly and are in the same order.")
        return True
    elif np.allclose(xyz1, xyz2):
        print("⚠️ Coordinates match approximately (e.g., due to float rounding).")
    else:
        print("❌ The coordinates do not match.")   
    return False 


class TreeExtractor:
    def __init__(self, input_path,  instance_field, semantic_field, output_dir=None, buffer=2.0):
        self.input_path = Path(input_path)
        self.instance_field = instance_field
        self.semantic_field = semantic_field
        self.buffer = buffer
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.coords = None
        # self.intensity = None
        self.semantic = None
        self.instance = None
        self.center = (0, 0, 0)

    def load_data(self):
        if self.input_path.suffix.lower() in [".laz", ".las"]:
            self.filetype = 'las'
            self._load_las()
        elif self.input_path.suffix.lower() == ".ply":
            self.filetype = 'ply'
            self._load_ply()
        else:
            raise ValueError(f"Unsupported file type: {self.input_path.suffix}")

    def _load_las(self):
        import laspy
        las = laspy.read(self.input_path)
        self.coords = las.xyz
        self.center = np.mean(self.coords, axis=0)
        self.coords -= self.center
        # self.intensity = las.intensity
        self.semantic = las[self.semantic_field]
        self.instance = las[self.instance_field]

        if 'red' in las.point_format.dimension_names:
            self.rgb = np.vstack((las.red, las.green, las.blue)).T.astype(np.uint16)
        else:
            self.rgb = self.assign_color()        

    def _load_ply(self):
        file_idx = self.input_path.stem.split("_")[-1]
        instance_path = self.input_path.parent / f"Instance_Results_forEval_{file_idx}.ply"
        semantic_path = self.input_path.parent / f"Semantic_results_forEval_{file_idx}.ply"

        instance_data = PlyData.read(instance_path)['vertex'].data
        semantic_data = PlyData.read(semantic_path)['vertex'].data

        # if not check_xyz_alignment(instance_data, semantic_data):
        #     raise ValueError

        self.coords = ply_to_xyz(instance_data)
        # self.center = np.mean(self.coords, axis=0)
        self.coords -= self.center
        # self.intensity = data['intensity']
        self.semantic = semantic_data['preds']
        self.instance = instance_data['preds'] + 1 # so that 0 is non-tree and tree starts with 1

        # assign color
        self.rgb = self._assign_color()

    def _assign_color(self):
        """assign color per tree"""
        unique_ids = np.unique(self.instance)
        colors = distinctipy.get_colors(len(unique_ids))
        if self.filetype == 'las':
            id_to_color = {tree_id: tuple(int(c * 65535) for c in color)
                           for tree_id, color in zip(unique_ids, colors)}
            return np.array([id_to_color[tid] for tid in self.instance], dtype=np.uint16)    
        elif self.filetype == 'ply':
            id_to_color = {tree_id: tuple(int(c * 255) for c in color)
                           for tree_id, color in zip(unique_ids, colors)}
            return np.array([id_to_color[tid] for tid in self.instance], dtype=np.uint8)

    def extract_trees(self):
        ids = np.unique(self.instance)
        unique_ids = ids[ids > 0]
        self.tree_neighbors = {}

        self._save_tree(tree_id=0) # output ground (non-tree points)

        for tree_id in tqdm(unique_ids, desc="Trees processed"):
            self._save_tree(tree_id)
            self.tree_neighbors[tree_id] = self._find_neighbors(tree_id)

        self._save_neighbor_table()

        print(f"All {len(unique_ids)} trees processed. Files in {self.output_dir}.")

    def _save_tree(self, tree_id):
        mask = (self.instance == tree_id)
        if not np.any(mask):
            return

        coords = self.coords[mask] + self.center
        colors = self.rgb[mask]
        dtype_fields = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                        ('red', 'u2'), ('green', 'u2'), ('blue', 'u2'),
                        # ('intensity', 'f4'), 
                        ('semantic_pred', 'i1'), ('instance_pred', 'i2')]
        array = np.empty(coords.shape[0], dtype=dtype_fields)
        array['x'] = coords[:, 0]
        array['y'] = coords[:, 1]
        array['z'] = coords[:, 2]
        array['red'] = colors[:, 0]
        array['green'] = colors[:, 1]
        array['blue'] = colors[:, 2]
        # array['intensity'] = self.intensity[mask]
        array['semantic_pred'] = self.semantic[mask]
        array['instance_pred'] = self.instance[mask]

        out_path = os.path.join(self.output_dir, f"{tree_id}.ply")
        ply_element = PlyElement.describe(array, 'vertex')
        PlyData([ply_element], text=False).write(out_path)
    
    def _find_neighbors(self, tree_id):
        mask = (self.instance == tree_id)
        tree_pts = self.coords[mask]
        if tree_pts.shape[0] == 0:
            return []

        mins = tree_pts.min(axis=0) - self.buffer
        maxs = tree_pts.max(axis=0) + self.buffer
        bbox_mask = np.all((self.coords >= mins) & (self.coords <= maxs), axis=1)
        trees_in_bbox = self.instance[bbox_mask]

        neighbor_ids = set(trees_in_bbox)
        neighbor_ids.discard(tree_id)
        neighbor_ids.discard(0)
        neighbor_ids.discard(-1)

        return sorted(neighbor_ids)
    
    def _save_neighbor_table(self):
        csv_path = os.path.join(self.output_dir, "tree_neighbors.csv")
        with open(csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["treeID", "neighbor_trees"])
            for tree_id, neighbors in self.tree_neighbors.items():
                neighbor_str = " ".join(str(n) for n in neighbors)
                writer.writerow([tree_id, neighbor_str])


def main():
    input_path = "outputs/TranCanadaHwy/prediction/2025-07-02_15/Instance_Results_forEval_0.ply"
    instance_field = "PredInstance"
    semantic_field = "PredSemantic"
    output_dir = "data/QC"

    extrator = TreeExtractor(input_path, instance_field, semantic_field, output_dir)
    extrator.load_data()
    extrator.extract_trees()


if __name__ == "__main__":
    main()