import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path


def main():
    input_path = "data/TranCanadaHwy/TranCanadaHwy.laz"
    input_path = Path(input_path)
    output_path = input_path.with_suffix('.ply')

    dtype_fields = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('intensity', 'f4'),
        ('semantic_seg', 'u1'),  # dummy label (e.g., 0)
        ('treeID', 'u4')         # dummy tree ID
        ]

    if input_path.suffix.lower() in [".laz", ".las"]:
        array = get_las(input_path, dtype_fields)
    elif input_path.suffix.lower() == ".ply":
        array = get_ply(input_path, dtype_fields)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
    array_to_ply(array, output_path)


def get_las(input_path, dtype_fields):
    las = laspy.read(input_path)
    x = las.x
    y = las.y
    z = las.z
    intensity = las.intensity

    n_points = len(x)

    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y
    array['z'] = z
    array['intensity'] = intensity
    array['semantic_seg'] = np.zeros(n_points, dtype='u1')
    array['treeID'] = np.zeros(n_points, dtype='u4')

    return array



def get_ply(input_path, dtype_fields):
    data = PlyData.read(input_path)
    vertex_data = data['vertex'].data

    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    intensity = vertex_data['scalar_Intensity']

    n_points = len(x)

    array = np.empty(n_points, dtype=dtype_fields)
    array['x'] = x
    array['y'] = y
    array['z'] = z
    array['intensity'] = intensity
    array['semantic_seg'] = np.zeros(n_points, dtype='u1')
    array['treeID'] = np.zeros(n_points, dtype='u4')

    return array


def array_to_ply(array, output_path):
    element = PlyElement.describe(array, 'vertex')
    PlyData([element], text=False).write(output_path)


if __name__ == "__main__":
    main()