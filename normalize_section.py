import cloudvolume as cv
import scipy
import json
import torch
import numpy as np

import time
import sys
import os

from copy import deepcopy

from helpers import get_np
from augment import set_up_transformation, generate_transform, apply_transform

start_section = int(sys.argv[1])
end_section = int(sys.argv[2])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[3]



################### MINNIE!!
img_dst_cv_path = 'gs://microns-seunglab/minnie_v4/processed/m6_normalized_test'
img_src_cv_path = 'gs://microns-seunglab/minnie_v4/raw'
resin_cv_path = 'gs://microns-seunglab/minnie_v3/alignment/sergiy_resin_mask'
resin_mip = 8
img_mip = 6

img_src_cv = cv.CloudVolume(img_src_cv_path, mip=img_mip, fill_missing=True,
                            progress=False,bounded=False,
                            parallel=1)

img_dst_cv = cv.CloudVolume(img_dst_cv_path, mip=img_mip, fill_missing=True,
                            progress=False,bounded=False,
                            parallel=10, info=deepcopy(img_src_cv.info),
                            non_aligned_writes=True)

img_dst_cv.info['data_type'] = 'float32'
img_dst_cv.commit_info()

if resin_cv_path is not None:
    resin_cv = cv.CloudVolume(resin_cv_path, mip=resin_mip, fill_missing=True,
                              progress=False, bounded=False,
                              parallel=1)

resin_scale_factor = 2**(resin_mip - img_mip)

cv_xy_start = [0, 0]
cv_xy_end = [8096, 8096]
patch_size = 8096 // 4
spoof_sample = {'src': None, 'tgt': None}

for z in range(start_section, end_section):
    print (z)
    s = time.time()
    cv_img_data = img_src_cv[cv_xy_start[0]:cv_xy_end[0], cv_xy_start[1]:cv_xy_end[1], z].squeeze()

    spoof_sample['src'] = cv_img_data
    spoof_sample['tgt'] = cv_img_data

    if resin_cv_path is not None:
        cv_resin_data = resin_cv[cv_xy_start[0]//resin_scale_factor:cv_xy_end[0]//resin_scale_factor,
                cv_xy_start[1]//resin_scale_factor:cv_xy_end[1]//resin_scale_factor,
                z].squeeze()

        cv_resin_data_ups = scipy.misc.imresize(cv_resin_data, resin_scale_factor*1.0)
        spoof_sample['src_plastic'] = cv_resin_data_ups
        spoof_sample['tgt_plastic'] = cv_resin_data_ups
    norm_transform = generate_transform(
        {img_mip: [{"type": "preprocess"},
            {"type": "sergiynorm",
             "mask_plastic": True,
             "low_threshold": -0.485,
             #"high_threshold": 0.35, #BASILLLL
             "high_threshold": 0.22, #MINNIE
             "filter_black": True,
             "bad_fill": -20.0}
                     ]}, img_mip, img_mip)
    norm_sample = apply_transform(spoof_sample, norm_transform)
    processed_patch = norm_sample['src'].squeeze()

    cv_processed_data = get_np(processed_patch.unsqueeze(2).unsqueeze(2)).astype(np.float32)

    print (z, np.mean(cv_img_data), np.mean(cv_processed_data))

    img_dst_cv[cv_xy_start[0]:cv_xy_end[0], cv_xy_start[1]:cv_xy_end[1], z] = cv_processed_data
    e = time.time()
    print (e - s, " sec")


