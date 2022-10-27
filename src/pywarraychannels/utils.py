import numpy as np

def water_filling(depth, volume, return_cells=False):
    depth_sorted = np.sort(depth)
    depth_sorted_min = depth_sorted[0]
    depth_sorted -= depth_sorted_min
    depth_sorted_cumsum = np.cumsum(depth_sorted)
    levels = (depth_sorted_cumsum + volume)/np.arange(1, len(depth_sorted) + 1)
    level_win = 0
    for level, depth_s in zip(levels, depth_sorted):
        if level > depth_s:
            level_win = level
        else:
            break
    level = level_win + depth_sorted_min
    if return_cells:
        return np.maximum(level - depth, 0)
    else:
        return level