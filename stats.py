
import fastremap
import numpy as np
import time
from helpers import coarsen_mask
import six

def get_patch_stats(src_cv, x, y, z, outputs,
        mask_cv=None, th_high=None, th_low=None):
    data = np.array(src_cv[x[0]:x[1], y[0]:y[1], z]).squeeze()

    if mask_cv is None:
        mask = np.zeros_like(data) > 0
    else:
        mask = np.array(mask_cv[x[0]:x[1], y[0]:y[1], z]).squeeze() > 0
    if th_low is not None:
        mask[data < th_low] = 0
    if th_high is not None:
        mask[data > th_high] = 0
    mask[data == 0] = 0

    masked_data = data[mask == False]

    result = {}
    if len(masked_data) > 50:
        for o in outputs:
            if o == 'mean':
                result[o] = np.mean(masked_data)
            elif o == 'var':
                result[o] = np.var(masked_data)
            else:
                raise Exception("Unkown stat to compute: {}".format(o))
    return result

def combine_patch_stats(patch_stats):
    num_patch = len(patch_stats)
    result = {}
    ps_count = 0

    for ps in patch_stats:
        for k, v in six.iteritems(ps):
            if k not in result:
                result[k] = v
            else:
                if k in ['mean', 'stdev']:
                    result[k] = (result[k] * ps_count + v) / (ps_count + 1)

        ps_count += 1
    return result

stats_info = {
  "data_type": "float32",
  "num_channels": 1,
  "scales": [
    {
      "chunk_sizes": [
        [
          1,
          1,
          1
        ]
      ],
      "encoding": "raw",
      "key": "4_4_40",
      "resolution": [
        4,
        4,
        40
      ],
      "size": [
        1024,
        1024,
        102400
      ],
      "voxel_offset": [
        0,
        0,
        0
      ]
    }
   ],

  "type": "image"
}


