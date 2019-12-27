import numpy as np

def calc_plane(x, y):
    a = np.column_stack((x, np.ones_like(x)))
    return np.linalg.lstsq(a, y)[0]
