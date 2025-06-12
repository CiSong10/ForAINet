import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.interpolate import bisplrep, bisplev
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from RANSAC.smallest_enclosing_circle import welzl
from utils.utils import (DTM_accuracy, DTM_generation, cal_DBH_and_centerP,
                         compute_volume_with_convex_hull, hdbscan_filtering, output_DTM_as_pc,)
# from utils.ply import read_ply, write_ply
# from tree_metrics.rmse import rmse


@dataclass
class ProcessingConfig:
    binsize: float = 0.5
    breast_height: float = 1.3
    min_points_threshold: int = 10
    iou_threshold: float = 0.5
    num_classes_sem: int = 6
    max_workers: int = 5
    # possible parameters: tree height threshold, tree dbh threshold


@dataclass
class TreeMetrics:
    height: Optional[float] = None
    trunk_diameter: Optional[float] = None
    crown_diameter: Optional[float] = None
    crown_volume: Optional[float] = None
    crown_volume_live: Optional[float] = None
    x_location: Optional[float] = None
    y_location: Optional[float] = None
    # possible pararmeters: leaning angle, live crown ratio


@dataclass
class PlotMetrics:
    dtm_coverage: Optional[float] = None
    dtm_rmse: Optional[float] = None
    stem_density: Optional[float] = None
    detection_rate: Optional[float] = None
    detection_accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    mean_iou: Optional[float] = None


def process_single_tree(args_tuple):
    (instance_id, tree_coords, ins_labels, sem_labels, interp_spline, plot_dir, tree_type, config) = args_tuple
    
    # Get points for this instance
    mask = (ins_labels == instance_id)
    x_pts = tree_coords['x'][mask]
    y_pts = tree_coords['y'][mask]
    z_pts = tree_coords['z'][mask].copy()
    sem_pts = sem_labels[mask]
    
    if len(x_pts) < config.min_points_threshold:
        return instance_id, None
    
    # Normalize height above ground
    center_x, center_y = np.mean(x_pts), np.mean(y_pts)
    ground_z = bisplev(center_x, center_y, interp_spline)
    z_pts -= ground_z
    
    # Filter points using HDBSCAN
    xyz_points = np.column_stack([x_pts, y_pts, z_pts])
    filtered_points, valid_indices = hdbscan_filtering(xyz_points)
    
    if len(filtered_points) < config.min_points_threshold:
        return instance_id, None
    
    # Calculate tree metrics
    metrics = TreeMetrics()
    metrics.height = float(np.max(filtered_points[:, 2]))
    metrics.x_location = float(center_x)
    metrics.y_location = float(center_y)

    # Output saving path
    im_path = str(plot_dir / f'{tree_type}Tree{instance_id}_fittingPointsProj.png')
    
    # Calculate DBH
    dbh, dbh_center_x, dbh_center_y = calculate_dbh(xyz_points, z_pts, sem_pts, instance_id, im_path, config)

    if dbh and dbh > 0:
        metrics.trunk_diameter = float(dbh)
        metrics.x_location = float(dbh_center_x)
        metrics.y_location = float(dbh_center_y)
    
    # Calculate crown metrics
    crown_diameter, crown_volume, crown_volume_live = calculate_crown_metrics(xyz_points, sem_pts, valid_indices, im_path)
    
    metrics.crown_diameter = float(crown_diameter) if crown_diameter > 0 else None
    metrics.crown_volume = float(crown_volume) if crown_volume > 0 else None
    metrics.crown_volume_live = float(crown_volume_live) if crown_volume_live > 0 else None
    
    return instance_id, metrics


def calculate_dbh(xyz_points: np.ndarray, z_pts: np.ndarray, sem_pts: np.ndarray,
                  instance_id: int, im_path: str, config) -> Tuple[float, float, float]:
    
    # Find stem points at breast height
    height_mask = (z_pts > (config.breast_height - 0.5)) & (z_pts < (config.breast_height + 0.5))
    stem_mask = (sem_pts == 2)  # 2 is the stem class label
    
    fitting_mask = height_mask & stem_mask
    
    # Expand search height range if not at least 10 points
    increment = 0.5
    while np.sum(fitting_mask) < config.min_points_threshold and increment < 4:
        increment += 0.2
        height_mask = (z_pts > (config.breast_height - increment)) & (z_pts < (config.breast_height + increment))
        fitting_mask = height_mask & stem_mask
    
    if np.sum(fitting_mask) < config.min_points_threshold:
        return None, np.mean(xyz_points[:, 0]), np.mean(xyz_points[:, 1])
    
    # Calculate DBH and create a dummy figure
    points_for_fitting = xyz_points[fitting_mask]

    fig = plt.figure()
    
    try:
        dbh, center_x, center_y = cal_DBH_and_centerP(points_for_fitting, fig, im_path, instance_id, config.min_points_threshold)
        plt.close(fig)
        return dbh, center_x, center_y
    except Exception as e:
        plt.close(fig)
        return None, np.mean(xyz_points[:, 0]), np.mean(xyz_points[:, 1])


def calculate_crown_metrics(xyz_points: np.ndarray, sem_pts: np.ndarray, valid_indices: np.ndarray, im_path: str) -> Tuple[float, float, float]:
    
    # Get live branch points (class 3) and all branch points (class 3 & 4)
    live_branch_mask = (sem_pts == 3) & np.isin(np.arange(len(sem_pts)), valid_indices)
    all_branch_mask = np.isin(sem_pts, [3, 4]) & np.isin(np.arange(len(sem_pts)), valid_indices)
    
    live_points = xyz_points[live_branch_mask]
    all_branch_points = xyz_points[all_branch_mask]
    
    if len(live_points) < 10 or len(all_branch_points) < 10:
        return 0, 0, 0
    
    # Calculate crown diameter using smallest enclosing circle
        # https://rosettacode.org/wiki/Smallest_enclosing_circle_problem#Python
        # Welzl's algorithm for smallest-circle problem
        # https://en.wikipedia.org/wiki/Smallest-circle_problem#Welzl's_algorithm    
    xy_points = all_branch_points[:, :2]
    try:
        nsphere = welzl(xy_points)
        crown_diameter = 2 * np.sqrt(nsphere.sqradius)
    except Exception:
        crown_diameter = 0
    
    # Calculate crown volumes
    try:
        crown_volume_live = compute_volume_with_convex_hull(live_points, im_path)
        crown_volume_total = compute_volume_with_convex_hull(all_branch_points, im_path)
    except Exception:
        crown_volume_live = 0
        crown_volume_total = 0
    
    return crown_diameter, crown_volume_total, crown_volume_live


class ForestryAnalyzer:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = self._setup_logging()    

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            # handlers=[logging.StreamHandler(),
            #           logging.FileHandler('forestry_analysis.log')],
        )
        return logging.getLogger(__name__)
    
    def process_all_plots(self, file_paths: List[str], input_dir: str, output_dir: str, test_file_paths: List[str], field_file_path: str) -> None:

        self.logger.info(f"Processing {len(file_paths)} plots")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process plots sequentially to avoid memory issues with large datasets
        # Parallelization happens at the tree level within each plot
        for idx, file_path in enumerate(file_paths):
            self._process_single_plot(idx, file_path, input_dir, output_path, test_file_paths, field_file_path)

    def _process_single_plot(self, plot_idx: int, file_path: str, input_dir: str, output_dir: Path, test_file_paths: List[str], field_file_path: str) -> None:
        self.logger.info(f"Processing plot {plot_idx}")

        plot_dir = output_dir / f"plot{file_path[-6:-4]}"
        plot_dir.mkdir(exist_ok=True)

        if (plot_dir / 'tree_analysis_results.csv').exists():
            self.logger.info(f"Plot {plot_idx} already processed, skipping")
            return

        sem_data, ins_data = self._load_plot_data(input_dir, file_path)
        if sem_data is None or ins_data is None:
            return
        
        coords, labels = self._extract_data(sem_data, ins_data)

        dtm_gt, dtm_pred = self._generate_dtms(coords, labels, plot_dir, file_path)
        
        plot_metrics = self._calculate_plot_metrics(coords, labels, dtm_gt, dtm_pred)
        
        gt_trees, pred_trees = self._process_trees(coords, labels, dtm_gt, dtm_pred, plot_dir)

        tree_metrics = self._match_and_evaluate_trees(gt_trees, pred_trees, plot_idx, test_file_paths, field_file_path)
        
        self._save_results(plot_dir, gt_trees, pred_trees, tree_metrics, plot_metrics, labels)

    def _load_plot_data(self, input_dir: str, filename: str) -> Tuple[Optional[Any], Optional[Any]]:
        sem_path = os.path.join(input_dir, filename)
        ins_path = os.path.join(input_dir, filename.replace('Semantic_results_forEval', 'Instance_Results_forEval'))
            
        sem_data = PlyData.read(sem_path)
        ins_data = PlyData.read(ins_path)
            
        return sem_data, ins_data

    def _extract_data(self, sem_data: Any, ins_data: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract coordinates and labels from loaded data."""
        coords = {'x': ins_data['vertex']['x'],
                  'y': ins_data['vertex']['y'],
                  'z': ins_data['vertex']['z'],}
        
        labels = {'pred_ins': ins_data['vertex']['preds'],
                  'pred_sem': sem_data['vertex']['preds'],
                  'gt_ins': ins_data['vertex']['gt'] - 1,
                  'gt_sem': sem_data['vertex']['gt'],}
        
        return coords, labels

    def _generate_dtms(self, coords: Dict, labels: Dict, plot_dir: Path, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        # Ground point indices
        idx_gt_ground = (labels['gt_sem'] == 1)
        idx_pred_ground = (labels['pred_sem'] == 1)
        
        # Generate DTMs
        dtm_gt = DTM_generation(coords['x'][idx_gt_ground], 
                                coords['y'][idx_gt_ground], 
                                coords['z'][idx_gt_ground], 
                                self.config.binsize)
                                
        dtm_pred = DTM_generation(coords['x'][idx_pred_ground], 
                                  coords['y'][idx_pred_ground], 
                                  coords['z'][idx_pred_ground], 
                                  self.config.binsize) if np.any(idx_pred_ground) else np.array([])
        
        # Save DTMs as point clouds
        plot_id = filename[-6:-4]
        output_DTM_as_pc(dtm_gt, plot_dir / f'floor_gt_pynn{plot_id}.ply')
        output_DTM_as_pc(dtm_pred, plot_dir / f'floor_pre_pynn{plot_id}.ply')
        
        return dtm_gt, dtm_pred
     
    def _calculate_plot_metrics(self, coords: Dict, labels: Dict, dtm_gt: np.ndarray, dtm_pred: np.ndarray) -> PlotMetrics:
        # DTM accuracy
        idx_gt_ground = (labels['gt_sem'] == 1)
        idx_pred_ground = (labels['pred_sem'] == 1)
        
        dtm_coverage, dtm_rmse = DTM_accuracy(coords, self.config.binsize, idx_gt_ground, idx_pred_ground)
        
        # Calculate stem density
        idx_gt_tree = np.isin(labels['gt_sem'], [2, 3, 4])
        tree_coords = np.column_stack([coords['x'][idx_gt_tree], coords['y'][idx_gt_tree]])
        
        if len(tree_coords) > 3:
            hull = ConvexHull(tree_coords)
            convex_area = hull.area
            unique_trees = len(np.unique(labels['gt_ins'][idx_gt_tree]))
            stem_density = (unique_trees / convex_area) * 10000  # trees per hectare
        else:
            stem_density = None
        
        return PlotMetrics(dtm_coverage=dtm_coverage,
                           dtm_rmse=dtm_rmse,
                           stem_density=stem_density,)

    def _process_trees(self, coords: Dict, labels: Dict, dtm_gt: np.ndarray, dtm_pred: np.ndarray, plot_dir: Path) -> Tuple[Dict, Dict]:
        
        idx_gt_tree = np.isin(labels['gt_sem'], [2, 3, 4])
        idx_pred_tree = np.isin(labels['pred_sem'], [2, 3, 4])
        
        gt_trees = self._process_tree_instances(coords, labels, idx_gt_tree, dtm_gt, 'gt', plot_dir)
        pred_trees = self._process_tree_instances(coords, labels, idx_pred_tree, dtm_pred, 'pred', plot_dir)
        
        return gt_trees, pred_trees

    def _process_tree_instances(self, coords: Dict, labels: Dict, tree_mask: np.ndarray, 
                                dtm: np.ndarray, tree_type: str, plot_dir: Path) -> Dict[int, TreeMetrics]:
        
        # Extract tree data
        tree_coords = {'x': coords['x'][tree_mask],
                       'y': coords['y'][tree_mask],
                       'z': coords['z'][tree_mask],}
        
        label_key = f'{tree_type}_ins'
        sem_key = f'{tree_type}_sem'
        
        tree_ins_labels = labels[label_key][tree_mask]
        tree_sem_labels = labels[sem_key][tree_mask]
        
        # Setup interpolation for ground height
        interp_spline = bisplrep(dtm[:, 0], dtm[:, 1], dtm[:, 2])
        
        tree_metrics = {}
        unique_instances = [i for i in np.unique(tree_ins_labels) if i != -1]

        unique_instances = unique_instances[:4] # select 4 instances for testing
        
        # Prepare arguments for parallel processing
        args_list = [(instance_id, tree_coords, tree_ins_labels, tree_sem_labels, interp_spline, plot_dir, tree_type, self.config) 
                     for instance_id in unique_instances]
        
        self.logger.info(f"Processing {len(unique_instances)} {tree_type} trees with {self.config.max_workers} workers")

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(process_single_tree, args): args[0] for args in args_list}

            for future in tqdm(as_completed(futures), total=len(futures), desc="trees"):
                instance_id = futures[future]
                try:
                    result = future.result()
                    if result[1]:
                        tree_metrics[instance_id] = result[1]
                except Exception as e:
                    self.logger.warning(f"Exception occurred while processing {tree_type} tree {instance_id}: {str(e)}")

        return tree_metrics

    def _match_and_evaluate_trees(self, gt_trees: Dict, pred_trees: Dict, plot_idx: int, 
                                  test_file_paths: List[str], field_file_path: str) -> Dict[str, List[float]]:
        
        # Load field data if available TODO: figure this out: how do I know if my trees are in field data?
        test_file = os.path.basename(test_file_paths[plot_idx]) # test_file_paths has to be in order of the matching_files
        field_dbh = self._load_field_data(test_file, list(gt_trees.keys()), field_file_path)
        
        # Match gt and pred trees using IoU
        matches = []
        for gt_id, gt_metrics in gt_trees.items():
            best_iou = 0.0
            best_pred_id = None
            
            for pred_id, pred_metrics in pred_trees.items():
                # Calculate IoU based on spatial overlap
                distance = np.sqrt((gt_metrics.x_location - pred_metrics.x_location)**2 + 
                                   (gt_metrics.y_location - pred_metrics.y_location)**2)
                
                # Simple IoU approximation based on distance and crown size, by comparing if distance < the sum of two radius of two trees
                overlap_threshold = max(gt_metrics.crown_diameter, pred_metrics.crown_diameter) / 2
                iou = max(0, 1 - distance / overlap_threshold) if overlap_threshold > 0 else 0
                
                if iou > best_iou and iou >= self.config.iou_threshold:
                    best_iou = iou
                    best_pred_id = pred_id
            
            if best_pred_id is not None:
                matches.append((gt_id, best_pred_id, best_iou))

        # Calculate metrics
        metrics = self._calculate_evaluation_metrics(gt_trees, pred_trees, matches, field_dbh)
        
        return metrics
    
    def _load_field_data(self, test_file: str, gt_labels: List[int], field_file_path: str) -> Dict[int, float]:
        """
        In the ply test_file (provided in the test_file_paths variable), look for tree labels that matches what we have in gt_trees. 
        And in the csv that share the same name as the ply test file, get the DBH of the trees.

        For example, if we have ground truth tree 2101, 2102, 2103, 2104;
        the test file is `data/data_set1_5classes/treeinsfused/raw/CULS/CULS_plot_2_annotated_test.ply`, 
        look inside `tree_metrics/field_gt/tree_data_CULS.csv`, find treeID = 2101, 2102, 2103, 2104; get their DBH

        return: field_dbh, a dictionary of {treeID: DBH}
        """
        field_dbh = {}
        
        try:
            # Extract plot information from filename to load the coorsponding filed data
            file_parts = Path(test_file).stem.split('_')
            dataset = file_parts[0] if file_parts else ""
            plot_id = None
            
            for part in file_parts:
                if part.isdigit():
                    plot_id = int(part)
                    break
            
            field_file =  Path(field_file_path) / f"tree_data_{dataset}.csv"

            # get field_dbh according to plotID and treeID
            
            if field_file.exists():
                df = pd.read_csv(field_file)
                
                for gt_label in gt_labels:
                    dbh_value = -1
                    
                    if 'plotID' in df.columns and plot_id is not None:
                        selected_row = df[(df['plotID'] == plot_id) & (df['treeID'] == gt_label)]
                    elif 'treeID' in df.columns:
                        selected_row = df[df['treeID'] == gt_label]
                    else:
                        continue
                    
                    if not selected_row.empty:
                        dbh_value = selected_row['DBH'].values[0]
                        # Convert from cm to m if necessary
                        if dbh_value > 1:
                            dbh_value /= 100
                        field_dbh[gt_label] = dbh_value
                        
        except Exception as e:
            self.logger.warning(f"Could not load field data: {e}")
        
        return field_dbh
    
    def _calculate_evaluation_metrics(self, gt_trees: Dict, pred_trees: Dict, 
                                    matches: List[Tuple], field_dbh: Dict) -> Dict[str, List[float]]:
        """
        `dbh_errors` are calculated using the DBH derived from the gt and pred point clouds.  
        `field_dbh_errors` are calculated by comparing the DBH from the gt point cloud with the DBH values in the field_gt CSV.
        """
        metrics = {'height_errors': [],
                   'dbh_errors': [],
                   'crown_diameter_errors': [],
                   'location_errors': [],
                   'crown_volume_errors': [],
                   'field_dbh_errors': []}
        
        if not matches:
            self.logger.warning(f"No matches found between ground truth and predicted trees")
            return metrics
        
        for gt_id, pred_id, iou in matches:
            gt_tree = gt_trees[gt_id]
            pred_tree = pred_trees[pred_id]
            
            if gt_tree.height > 0 and pred_tree.height > 0:
                metrics['height_errors'].append(pred_tree.height - gt_tree.height)
            
            if gt_tree.trunk_diameter > 0 and pred_tree.trunk_diameter > 0:
                metrics['dbh_errors'].append(pred_tree.trunk_diameter - gt_tree.trunk_diameter)
            
            if gt_tree.crown_diameter > 0 and pred_tree.crown_diameter > 0:
                metrics['crown_diameter_errors'].append(pred_tree.crown_diameter - gt_tree.crown_diameter)
            
            location_error = np.sqrt((pred_tree.x_location - gt_tree.x_location)**2 + 
                                     (pred_tree.y_location - gt_tree.y_location)**2)
            metrics['location_errors'].append(location_error)
            
            if gt_tree.crown_volume > 0 and pred_tree.crown_volume > 0:
                metrics['crown_volume_errors'].append(pred_tree.crown_volume - gt_tree.crown_volume)
            
            if gt_id in field_dbh and field_dbh[gt_id] > 0 and pred_tree.trunk_diameter > 0:
                metrics['field_dbh_errors'].append(pred_tree.trunk_diameter - field_dbh[gt_id])
        return metrics
    
    def _save_results(self, plot_dir: Path, gt_trees: Dict, pred_trees: Dict, 
                     tree_metrics: Dict, plot_metrics: PlotMetrics, labels: Dict) -> None:
        self._save_tree_results(plot_dir, gt_trees, pred_trees, tree_metrics)
        self._save_plot_results(plot_dir, plot_metrics, tree_metrics, len(gt_trees), len(pred_trees))
        self._save_semantic_results(plot_dir, labels)
    
    def _save_tree_results(self, plot_dir: Path, gt_trees: Dict, pred_trees: Dict, metrics: Dict) -> None:
        
        tree_data = []
        
        for gt_id, gt_tree in gt_trees.items():
            row = {
                'gt_label': gt_id,
                'gt_height': gt_tree.height,
                'gt_dbh': gt_tree.trunk_diameter,
                'gt_crown_diameter': gt_tree.crown_diameter,
                'gt_crown_volume': gt_tree.crown_volume,
                'gt_x_location': gt_tree.x_location,
                'gt_y_location': gt_tree.y_location
            }
            tree_data.append(row)
        
        df = pd.DataFrame(tree_data)
        
        summary_stats = self._calculate_summary_statistics(metrics)
        summary_df = pd.DataFrame([summary_stats])

        df.to_csv(plot_dir / 'tree_analysis_results.csv')
        summary_df.to_csv(plot_dir / 'summary_statistics.csv')
    
    def _calculate_summary_statistics(self, metrics: Dict) -> Dict[str, float]:
        stats = {}
        
        for metric_name, errors in metrics.items():
            if errors:
                errors_array = np.array(errors)
                stats[f'{metric_name}_rmse'] = np.sqrt(np.mean(errors_array**2))
                stats[f'{metric_name}_bias'] = np.mean(errors_array)
                stats[f'{metric_name}_std'] = np.std(errors_array)
        
        return stats
    
    def _save_plot_results(self, plot_dir: Path, plot_metrics: PlotMetrics, 
                          tree_metrics: Dict, n_gt_trees: int, n_pred_trees: int) -> None:
        
        n_matched = len([errors for errors in tree_metrics['height_errors']])
        detection_rate = (n_pred_trees / n_gt_trees * 100) if n_gt_trees > 0 else 0
        detection_accuracy = (n_matched / n_gt_trees * 100) if n_gt_trees > 0 else 0
        
        plot_results = {
            'dtm_coverage': plot_metrics.dtm_coverage,
            'dtm_rmse': plot_metrics.dtm_rmse,
            'stem_density': plot_metrics.stem_density,
            'n_reference_trees': n_gt_trees,
            'n_detected_trees': n_pred_trees,
            'n_matched_trees': n_matched,
            'detection_rate': detection_rate,
            'detection_accuracy': detection_accuracy,
            'omission_error': 100 - detection_accuracy,
            'commission_error': ((n_pred_trees - n_matched) / n_pred_trees * 100) if n_pred_trees > 0 else 0
        }
        
        with open(plot_dir / 'plot_metrics.json', 'w') as f:
            json.dump(plot_results, f, indent=2)
    
    def _save_semantic_results(self, plot_dir: Path, labels: Dict) -> None:
        
        pred_sem = labels['pred_sem'] + 1
        gt_sem = labels['gt_sem'] + 1
        
        n_classes = self.config.num_classes_sem

        # Compute confusion matrix with all 6 classes included (0 unclassified is often not in pred_sem nor gt_sem)
        cm = confusion_matrix(gt_sem, pred_sem, labels=list(range(n_classes)))
        
        # Calculate IoU for each class
        iou_per_class = []
        for i in range(n_classes):
            if cm[i, i] == 0:
                iou_per_class.append(0.0)
            else:
                iou = cm[i, i] / (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                iou_per_class.append(iou)
        
        # Save semantic results
        semantic_results = {
            'overall_accuracy': np.trace(cm) / np.sum(cm),
            'mean_accuracy': np.mean([cm[i, i] / cm[i, :].sum() for i in range(n_classes) if cm[i, :].sum() > 0]),
            'mean_iou': np.mean([iou for iou in iou_per_class if iou > 0]),
            'iou_per_class': iou_per_class
        }
        
        with open(plot_dir / 'semantic_metrics.json', 'w') as f:
            json.dump(semantic_results, f, indent=2)


def main():
    config = ProcessingConfig()

    input_dir = 'outputs/pretrained/eval/2025-05-27_15-38-23' 
    output_dir = os.path.join(input_dir, 'para_cal_imgs')
    test_file_paths = [
        'data/data_set1_5classes/treeinsfused/raw/CULS/CULS_plot_2_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_1_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/TUWIEN/TUWIEN_test_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_17_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_18_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_22_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_23_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/NIBIO/NIBIO_plot_5_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/RMIT/RMIT_test_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/SCION/SCION_plot_31_annotated_test.ply', 
        'data/data_set1_5classes/treeinsfused/raw/SCION/SCION_plot_61_annotated_test.ply'
        ]
    field_file_path = "tree_metrics/field_gt"

    all_files = os.listdir(input_dir) # all_files gets a list in arbitary order
    matching_files = [f for f in all_files if 'Semantic_results_forEval' in f]
    
    analyzer = ForestryAnalyzer(config)
    analyzer.process_all_plots(matching_files, input_dir, output_dir, test_file_paths, field_file_path)
 
    print("Processing completed!")


if __name__ == "__main__":
    main()