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

#start_section = int(sys.argv[1])
#end_section = int(sys.argv[2])

print ("Kreating source cloud volume")
cv_src = cv.CloudVolume(cv_src_path, mip=mip_in, fill_missing=True, bounded=False, progress=False,
        parallel=32)
print ("Kreating dst cloud volume")
cv_dst = cv.CloudVolume(cv_dst_path, mip=img_out_mip,
                        fill_missing=True, bounded=False,
                        progress=False, parallel=32, non_aligned_writes=True)

src_cv_start = [0, 8192, 300000]
tgt_cv_start = [0, 0, 3000]

tgt_step_sizes = [1024*16, 1024*16, 1]
src_step_sizes = [1024*16, 1024*16, 100]

tgt_chunk_sizes = tgt_step_sizes
src_chunk_sizes = tgt_chunk_sizes

src_cv_end = [231424, 114688, 300300]
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

if __name__ == "__main__":
    parser.add_argument('--mode', type=str)
    parser.add_argument('--queue_name', type=str)
    parser.add_argument('--cv_src_path', type=str)
    parser.add_argument('--cv_dst_path', type=str)

    parser.add_argument('--bbox_start', type=str, default=None)
    parser.add_argument('--bbox_end', type=str, default=None)
    parser.add_argument('--mip', type=int, default=0)

    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--lease_seconds', type=int, default=60)
    parser.add_argument('--processor_name', '-p', type=str)
    parser.add_argument('--processor_args', '-a', type=str)

    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    print ("Hello, world!")


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    copy_info = True
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

    with TaskQueue(args.queue_name) as tq:
        if args.mode == "worker":
            work(tq, args.lease_seconds)
        elif args.mode == "master":
            # TODO: proper argument parsing
            bbox_start = [int(i) for i in args.bbox_start.split(',')]
            bbox_end = [int(i) for i in args.bbox_end.split(',')]
            for z in range(bbox_start[-1], bbox_end[-1]):
                tq.insert(
                    ProcessorTask(
                        mip=args.mip,
                        crop=args.crop,
                        bbox_start=[bbox_start[0], bbox_start[1], z],
                        bbox_end=[bbox_end[0], bbox_end[1], z+1],
                        path =args.cv_path,
                        processor_name=args.processor_name,
                        processor_args=args.processor_args,
                        suffix=args.suffix
                    )
                )
