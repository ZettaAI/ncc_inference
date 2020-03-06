import os
import sys
import time
import json
import copy

import cloudvolume as cv
import numpy as np
from scipy.ndimage.measurements import label
from cloudvolume.lib import Bbox

import torch
from helpers import get_np
from taskqueue import RegisteredTask, TaskQueue
from tissue_mask import tissue_mask_detector_cv
from misalignment_mask import misalignment_mask_detector_cv

import argparse

def processor_lookup(processor_name):
    lookup_map = {
        'tissue_mask': tissue_mask_detector_cv,
        'misalignment_mask': misalignment_mask_detector_cv
    }
    return lookup_map[processor_name]


class ProcessorTask(RegisteredTask):
    def __init__(self, bbox_start, bbox_end, path, mip, processor_name,
            processor_args={}, patch_size=3072, crop=256, suffix=None):
        super().__init__(bbox_start, bbox_end, path, mip, processor_name,
                processor_args, patch_size, crop, suffix)

        '''self.bbox_start = [int(x) for x in bbox_start]
        self.bbox_end = [int(x) for x in bbox_end]
        self.path = path
        self.mip
        self.processor_name = processor_name
        self.size =
        self.suffix = suffix
        self.patch = patch
        self.crop = crop'''

    def execute(self):
        if self.bbox_start and self.bbox_end:
            apply_processor_3dbbox(self.bbox_start, self.bbox_end, self.path,
                    self.mip, self.processor_name, self.processor_args,
                    self.patch_size, self.crop, self.suffix)
        else:
            print(self)

def apply_processor_3dbbox(bbox_start, bbox_end, path, mip, processor_name,
                           processor_args, patch_size, crop, suffix):
    s = time.time()
    processor = processor_lookup(processor_name)
    src_cv_path = path

    if suffix is None:
        suffix = processor_name
    assert len(suffix) > 0
    dst_cv_path = os.path.join(
        path, suffix
    )

    src_cv = cv.CloudVolume(
        src_cv_path,
        mip=mip,
        fill_missing=True,
        bounded=False,
        progress=False,
    )
    dst_cv = cv.CloudVolume(
        dst_cv_path,
        mip=mip,
        fill_missing=True,
        bounded=False,
        progress=False,
        parallel=1,
        info=copy.deepcopy(src_cv.info),
        non_aligned_writes=False,
        delete_black_uploads=False,
        autocrop=True,
    )
    dst_cv.info["data_type"] = "float32"
    #dst_cv.commit_info()

    cv_xy_start = [0, 0]

    bbox = Bbox([i // 2**mip for i in bbox_start[0:2]] + bbox_start[2:3],
            [i // 2**mip for i in bbox_end[0:2]] + bbox_end[2:3])
    aligned_bbox = bbox.expand_to_chunk_size(src_cv.chunk_size,
                                             src_cv.voxel_offset)
    start_section = bbox_start[2]
    end_section = bbox_end[2]

    xy_start = aligned_bbox.minpt[0:2]
    xy_end = aligned_bbox.maxpt[0:2]
    print ("Writing to {}".format(dst_cv_path))
    for z in range(start_section, end_section):
        print ("SECTION {}".format(z))
        for i in range(0, (xy_end[0] - xy_start[0] + patch_size - 1)//patch_size):
            for j in range(0, (xy_end[1] - xy_start[1] + patch_size - 1)//patch_size):
                x = [xy_start[0] + i*patch_size, xy_start[0] + (i + 1) * patch_size]
                y = [xy_start[1] + j*patch_size, xy_start[1] + (j + 1) * patch_size]
                x_pad = copy.deepcopy(x)
                y_pad = copy.deepcopy(y)
                if i != 0:
                    x_pad[0] -= crop
                if j != 0:
                    y_pad[0] -= crop
                if i < ((xy_end[0] - xy_start[0] + patch_size - 1)//patch_size - 1) :
                    x_pad[1] += crop
                if j < ((xy_end[1] - xy_start[1] + patch_size - 1)//patch_size - 1):
                    y_pad[1] += crop
                print (x, y)

                #curr_chunk = src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                #prev_chunk = src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                result = processor(src_cv, x_pad, y_pad, z, **(json.loads(processor_args))).astype(np.float32)
                print (result.shape)
                if i != 0:
                    result = result[..., crop:, :]
                if j != 0:
                    result = result[..., :, crop:]
                if i < ((xy_end[0] - xy_start[0] + patch_size - 1)//patch_size - 1):
                    result = result[..., :-crop, :]
                if j < ((xy_end[1] - xy_start[1] + patch_size - 1)//patch_size - 1):
                    result = result[..., :, :-crop]

                dst_cv[x[0]:x[1], y[0]:y[1], z] = result[..., np.newaxis, np.newaxis]

    e = time.time()
    print(e - s, " sec")

def work(tq, lease_seconds=60):
    print ("start polling tasks...")
    tq.poll(lease_seconds=int(lease_seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train everything.')

    parser.add_argument('--mode', type=str)
    parser.add_argument('--queue_name', type=str)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--cv_path', type=str, default=None)
    parser.add_argument('--bbox_start', type=str, default=None)
    parser.add_argument('--bbox_end', type=str, default=None)
    parser.add_argument('--mip', type=int, default=4)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--lease_seconds', type=int, default=60)
    parser.add_argument('--processor_name', '-p', type=str)
    parser.add_argument('--processor_args', '-a', type=str)

    args = parser.parse_args()
    print ("Hello, world!")

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
