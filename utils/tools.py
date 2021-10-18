import numpy as np

def resize_maps(map,factor):
    """Resize the maps by rescaling
    Args:
        map:0/1 array indicating obstacle locations
        factor: factor by which to rescaling the environment
    """
    (height,width) = map.shape
    row_indices = np.array(i for i in range(height) for _ in range(factor))
    col_indices = np.array(i for i in range(width) for _ in range(factor))
    walls = map[row_indices]
    walls = map[:,col_indices]
    assert walls.shape == (factor*height,factor*width)
    return map