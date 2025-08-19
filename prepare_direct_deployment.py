"""
This script prepares data for direct deployment, by adding dummy `semantic_seg` and `treeID` columns to the data.
"""

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path


def main():
    input_path = "data/TranCanadaHwy/TranCanadaHwy.laz"
    input_path = Path(input_path)
    # output_path = input_path.with_suffix('.ply')
    output_path = input_path.parent / "new_.ply"
    center_filename = input_path.parent / "center.txt"

    dtype_fields = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('intensity', 'f4'), # float32. Intensity typically scaled to values of 0–1 (float), 0–255 (int8), or 0–65,535 (int16), depending on the manufacturer
        ('semantic_seg', 'i1'),  # int8 — supports -1 and up to 127 classes
        ('treeID', 'i2')         # int16 — supports -1 and up to 32k trees
        ]

    if input_path.suffix.lower() in [".laz", ".las"]:
        array = get_las(input_path, dtype_fields, center_filename)
    elif input_path.suffix.lower() == ".ply":
        array = get_ply(input_path, dtype_fields, center_filename)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
    array_to_ply(array, output_path)


def get_las(input_path, dtype_fields, center_filename = "center.txt"):
    las = laspy.read(input_path)
    x = las.x
    y = las.y
    z = las.z
    intensity = las.intensity

    x, y = recenter_point_cloud(x, y, center_filename)
    n_points = len(x)

    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y
    array['z'] = z
    array['intensity'] = intensity
    array['semantic_seg'] = np.zeros(n_points, dtype='u1')
    array['treeID'] = np.zeros(n_points, dtype='u4')

    return array


def get_ply(input_path, dtype_fields, center_filename = "center.txt"):
    ply_data = PlyData.read(input_path)
    vertex_data = ply_data['vertex'].data

    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    intensity = vertex_data['intensity']

    x, y = recenter_point_cloud(x, y, center_filename)
    n_points = len(x)

    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y
    array['z'] = z
    array['intensity'] = intensity
    array['semantic_seg'] = np.zeros(n_points, dtype='u1')
    array['treeID'] = np.zeros(n_points, dtype='u4')

    return array


def recenter_point_cloud(x, y, center_filename = "center.txt"):
    center = np.mean([x, y], axis=1)
    x -= center[0]
    y -= center[1]
    np.savetxt(center_filename, center.reshape(1, 2), header='x_center y_center')
    return x, y


def array_to_ply(array, output_path):
    element = PlyElement.describe(array, 'vertex')
    PlyData([element], text=False).write(output_path)


if __name__ == "__main__":
    main()