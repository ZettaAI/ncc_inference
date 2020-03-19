import cloudvolume as cv
import scipy
import json
import torch
import numpy as np
import time
from copy import deepcopy

import artificery
import copy
import os
import sys
from pdb import set_trace as st

from tqdm import tqdm
import itertools
from math import ceil

gpu_id = '0'
if len(sys.argv) > 3:
    gpu_id = sys.argv[3]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

#start_section = int(sys.argv[1])
#end_section = int(sys.argv[2])
copy_info = True
cv_src_path ='gs://fafb_v15_montages/FAFB_montage/v15_montage_20190901_rigid_split/1024x1024'
cv_dst_path = 'gs://fafb_v15_montages/sergiy_playground/test1'
mip_in = mip_out = 0

print ("Kreating source cloud volume")
cv_src = cv.CloudVolume(cv_src_path, mip=mip_in, fill_missing=True, bounded=False, progress=False,
        parallel=32)
print ("Kreating dst cloud volume")
if copy_info:
    cv_dst = cv.CloudVolume(cv_dst_path, mip=mip_out,
                            fill_missing=True, bounded=False,
                            progress=False, parallel=32,
                            info=deepcopy(cv_src.info), non_aligned_writes=True)
    cv_dst.info['scales'][0]['key'] = '4_4_40'
    cv_dst.info['scales'][0]['resolution'] = [4, 4, 40]

    cv_dst.commit_info()
else:
    cv_dst = cv.CloudVolume(cv_dst_path, mip=img_out_mip,
                            fill_missing=True, bounded=False,
                            progress=False, parallel=32, non_aligned_writes=True)
src_cv_start = [0, 8192, 330000]
tgt_cv_start = [0, 0, 3300]

tgt_step_sizes = [1024*16, 1024*16, 1]
src_step_sizes = [1024*16, 1024*16, 100]

tgt_chunk_sizes = tgt_step_sizes
src_chunk_sizes = tgt_chunk_sizes

src_cv_end = [231424, 114688, 341000]
tgt_cv_end = [tgt_cv_start[i] + \
              ceil((src_cv_end[i] - src_cv_start[i]) / (src_step_sizes[i] / tgt_step_sizes[i])) \
              for i in range(3)]

step_counts = [ceil((src_cv_end[i] - src_cv_start[i]) / src_step_sizes[i]) for i in range(3)]
iteration_ranges = [list(range(step_counts[i])) for i in range(3)]
chunk_ids = list(itertools.product(*iteration_ranges))

print (tgt_cv_start, tgt_cv_end)
print (src_cv_start, src_cv_end)
print (step_counts)
print (iteration_ranges)
print (len(list(chunk_ids)))

s = time.time()
for chunk_id in tqdm(chunk_ids):
    src_chunk_coords = [(src_cv_start[i] + chunk_id[i] * src_step_sizes[i],
                         src_cv_start[i] + chunk_id[i] * src_step_sizes[i] + src_chunk_sizes[i]) \
                        for i in range(3)]
    tgt_chunk_coords = [(tgt_cv_start[i] + chunk_id[i] * tgt_chunk_sizes[i],
                         tgt_cv_start[i] + (chunk_id[i] + 1) * tgt_chunk_sizes[i]) \
                        for i in range(3)]

    cv_src_data = cv_src[src_chunk_coords[0][0]:src_chunk_coords[0][1],
                         src_chunk_coords[1][0]:src_chunk_coords[1][1],
                         src_chunk_coords[2][0]:src_chunk_coords[2][1]]
    cv_dst[tgt_chunk_coords[0][0]:tgt_chunk_coords[0][1],
           tgt_chunk_coords[1][0]:tgt_chunk_coords[1][1],
           tgt_chunk_coords[2][0]:tgt_chunk_coords[2][1]] = cv_src_data

e = time.time()
print (e - s, " sec")
