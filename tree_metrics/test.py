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
from utils.debug import save_pickle, load_pickle
import pickle


def main():
    config = ProcessingConfig()

    input_dir = 'outputs/TranCanadaHwy_ForestSens' 
    output_dir = os.path.join(input_dir, 'para_cal_imgs')

    all_files = os.listdir(input_dir) # all_files gets a list in arbitary order
    matching_files = [f for f in all_files] # Adjust file filter
    
    analyzer = ForestryAnalyzer(config)
    analyzer.process_all_plots(matching_files, input_dir, output_dir)

    import laspy

    for filename in matching_files:
        data_path = os.path.join(input_dir, filename)
        data = laspy.read(data_path)
        points = data.points
        print("Hello")


if __name__ == "__main__":
    main()
