import cc3d
import fastremap
import numpy as np
import time
from helpers import coarsen_mask

def filter_conn_will(array, size):
    s = time.time()
    cc_labels = cc3d.connected_components(array)
    segids, counts = np.unique(cc_labels, return_counts=True)
    segids = [ segid for segid, ct in zip(segids, counts) if
            (ct > size and segid != 0)]
    cc_labels = fastremap.mask_except(cc_labels, segids, in_place=True)
    print ("cc time: {}".format(time.time() - s))
    return cc_labels

def tissue_mask_detector_cv(src_cv, x, y, z, th, size):
    data = np.array(src_cv[x[0]:x[1], y[0]:y[1], z]).squeeze()
    result = tissue_mask_detector(data, th, size)
    return result

def tissue_mask_detector(data, th, size):
    black = data == 0
    big_black = filter_conn_will(black, 10000) > 0
    black_coarse = coarsen_mask(big_black, 11)

    data_th = (data != 0) * (data > float(th[0])) * (data < float(th[1]))
    data_th_bl = data_th * (black_coarse == False)
    data_ft = filter_conn_will(data_th_bl, size) > 0
    print ((data != 0).sum(), data_th.sum(), (data > float(th[0])).sum(),
            (data < float(th[1])).sum(), data_ft.sum())
    return data_ft



