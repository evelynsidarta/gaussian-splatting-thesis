import pathlib
import OpenEXR as exr
import Imath
import numpy as np

# used to extract numpy array from gt directory
def read_depth_exr_file(filepath: pathlib.Path):
    
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))

    # normalized
    # depth_min = depth_map.min()
    # depth_max = depth_map.max()
    # max_val = (2**(8*2)) - 1
    
    # if depth_max - depth_min > np.finfo("float").eps:
    #     depth_map = max_val * (depth_map - depth_min) / (depth_max - depth_min)
    # else:
    #     depth_map = np.zeros(depth_map.shape, dtype=depth_map.dtype)

    # depth_map = -1 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    # depth_map = depth_map + 1

    print(depth_map)

    return depth_map