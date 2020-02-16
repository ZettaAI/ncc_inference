import os
import sys
import time
from copy import deepcopy

import cloudvolume as cv
import numpy as np
from cloudvolume.lib import Bbox

import torch
from helpers import get_np
from taskqueue import RegisteredTask, TaskQueue
from misalignment import misalignment_detector

import argparse

class DetectMisalignmentTask(RegisteredTask):
    def __init__(self, bbox_start, bbox_end, img_src_cv_path, mip=4):
        #resin_cv_path = os.path.join(img_src_cv_path, "resin_mask")
        img_dst_cv_path = os.path.join(img_src_cv_path, "miss_alignment_so_much")
        super(DetectMisalignmentTask, self).__init__(bbox_start, bbox_end, img_src_cv_path, mip)

        # attributes passed to super().__init__ are automatically assigned
        # use this space to perform additional processing such as:

        self.img_dst_cv_path = img_dst_cv_path
        self.img_src_cv_path = img_src_cv_path
        self.bbox_start = [int(x) for x in bbox_start]
        self.bbox_end = [int(x) for x in bbox_end]
        self.mip = mip

    def execute(self):
        if self.bbox_start and self.bbox_end:
            detect_misalignment_3d_bbox(
                self.bbox_start,
                self.bbox_end,
                self.img_dst_cv_path,
                self.img_src_cv_path,
                self.mip
            )
        else:
            print(self)


def detect_misalignment_3d_bbox(
        bbox_start, bbox_end, img_dst_cv_path, img_src_cv_path, mip,
        patch_size=2048

):

    s = time.time()
    img_src_cv = cv.CloudVolume(
        img_src_cv_path,
        mip=mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        parallel=1,
    )

    img_dst_cv = cv.CloudVolume(
        img_dst_cv_path,
        mip=mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        parallel=1,
        info=deepcopy(img_src_cv.info),
        autocrop=True,
        delete_black_uploads=False,
    )

    #img_dst_cv.info["data_type"] = "float32"
    img_dst_cv.commit_info()
    bbox = Bbox([i // 2**mip for i in bbox_start[0:2]] + bbox_start[2:3],
            [i // 2**mip for i in bbox_end[0:2]] + bbox_end[2:3])
    aligned_bbox = bbox.expand_to_chunk_size(img_src_cv.chunk_size,
                                             img_src_cv.voxel_offset)
    start_section = bbox_start[2]
    end_section = bbox_end[2]

    xy_start = aligned_bbox.minpt[0:2]
    xy_end = aligned_bbox.maxpt[0:2]

    for z in range(start_section, end_section):
        print ("SECTION {}".format(z))
        for i in range(0, (xy_end[0] - xy_start[0] + patch_size - 1)//patch_size):
            for j in range(0, (xy_end[1] - xy_start[1] + patch_size - 1)//patch_size):
                x = [xy_start[0] + i*patch_size, xy_start[0] + (i + 1) * patch_size]
                y = [xy_start[1] + j*patch_size, xy_start[1] + (j + 1) * patch_size]
                print (x, y)

                curr_chunk = img_src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                prev_chunk = img_src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                fcorr = misalignment_detector(torch.Tensor(curr_chunk).cuda(),
                                          torch.Tensor(prev_chunk).cuda(),
                                          mip=mip).squeeze()
                img_dst_cv[x[0]:x[1], y[0]:y[1], z] = fcorr[..., np.newaxis, np.newaxis]

    e = time.time()
    print(e - s, " sec")


def work(tq):
    tq.poll(lease_seconds=int(300))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train everything.')

    parser.add_argument('--mode', type=str)
    parser.add_argument('--queue_name', type=str)
    parser.add_argument('--cv_path', type=str, default=None)
    parser.add_argument('--bbox_start', type=str, default=None)
    parser.add_argument('--bbox_end', type=str, default=None)
    parser.add_argument('--mip', type=int, default=4)

    args = parser.parse_args()


    with TaskQueue(args.queue_name) as tq:
        if args.mode == "worker":
            work(tq)
        elif args.mode == "master":
            # TODO: proper argument parsing
            bbox_start = [int(i) for i in args.bbox_start.split(',')]
            bbox_end = [int(i) for i in args.bbox_end.split(',')]
            for z in range(bbox_start[-1], bbox_end[-1]):
                tq.insert(
                    DetectMisalignmentTask(
                        mip=args.mip,
                        bbox_start=[bbox_start[0], bbox_start[1], z],
                        bbox_end=[bbox_end[0], bbox_end[1], z+1],
                        img_src_cv_path=args.cv_path
                    )
                )
