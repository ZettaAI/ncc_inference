import numpy as np
import time

def copy_cv(src_cv, x, y, z):
    data = np.array(src_cv[x[0]:x[1], y[0]:y[1], z]).squeeze()
    return data



