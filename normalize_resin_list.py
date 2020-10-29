import os
import sys
import time
from copy import deepcopy

import cloudvolume as cv
import numpy as np
from PIL import Image

import torch
from augment import apply_transform, generate_transform
from helpers import get_np
from taskqueue import RegisteredTask, TaskQueue


def read_resin_paths(resin_text_path):
    resin_cv_paths = []
    resin_thresholds = []
    with open(resin_text_path, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(',')
            resin_cv_paths.append(items[0])
            resin_thresholds.append(float(items[1]))
            line = f.readline()
    return resin_cv_paths, resin_thresholds


class NormalizeTask(RegisteredTask):
    def __init__(self, start_section, end_section, img_src_cv_path, resin_text_path):
        img_dst_cv_path = os.path.join(img_src_cv_path, "m6_normalized")
        super(NormalizeTask, self).__init__(start_section, end_section, img_src_cv_path, resin_text_path)

        # attributes passed to super().__init__ are automatically assigned
        # use this space to perform additional processing such as:
        self.start_section = int(start_section)
        self.end_section = int(end_section)

        self.img_dst_cv_path = img_dst_cv_path
        self.img_src_cv_path = img_src_cv_path
        self.resin_cv_paths, self.resin_thresholds = read_resin_paths(resin_text_path)

    def execute(self):
        if self.start_section and self.end_section:
            normalize_section_range(
                self.start_section,
                self.end_section,
                self.img_dst_cv_path,
                self.img_src_cv_path,
                self.resin_cv_paths,
                self.resin_thresholds
            )
        else:
            print(self)


def normalize_section_range(
    start_section, end_section, img_dst_cv_path, img_src_cv_path, resin_cv_paths, resin_thresholds
):
    resin_mip = 6
    img_mip = 6

    resin_cvs = []
    for cv_path in resin_cv_paths:
        resin_cvs.append(cv.CloudVolume(
            cv_path,
            mip=resin_mip,
            fill_missing=True,
            progress=False,
            bounded=False,
            parallel=1,
        ))

    img_src_cv = cv.CloudVolume(
        img_src_cv_path,
        mip=img_mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        parallel=1,
    )

    # FIXME: Will not work. Need to run 1 section first creating and
    # committing the info file. However, then you need put the comments back
    # if you are running this in a distributed manner.
    # The fix should be to create the CV in master as opposed to worker.
    img_dst_cv = cv.CloudVolume(
        img_dst_cv_path,
        mip=img_mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        # info=copy.deepcopy(cv_src.info),
        parallel=1,
        non_aligned_writes=False,
        autocrop=True,
        delete_black_uploads=True,
    )

    # img_dst_cv.info["data_type"] = "float32"
    # img_dst_cv.commit_info()

    resin_scale_factor = 2 ** (resin_mip - img_mip)

    cv_xy_start = [0, 0]
    cv_xy_end = [8192, 8192]
    spoof_sample = {"src": None, "tgt": None}

    for z in range(start_section, end_section):
        print(z)
        s = time.time()
        cv_img_data = img_src_cv[
            cv_xy_start[0] : cv_xy_end[0], cv_xy_start[1] : cv_xy_end[1], z
        ].squeeze()

        spoof_sample["src"] = cv_img_data
        spoof_sample["tgt"] = cv_img_data

        if resin_cv_paths is not None:
            cv_resin_data = None
            for i in range(len(resin_cvs)):
                resin_cv = resin_cvs[i]
                resin_threshold = resin_thresholds[i]
                cur_data = resin_cv[
                    cv_xy_start[0]
                    // resin_scale_factor : cv_xy_end[0]
                    // resin_scale_factor,
                    cv_xy_start[1]
                    // resin_scale_factor : cv_xy_end[1]
                    // resin_scale_factor,
                    z,
                ].squeeze()
                cur_data[cur_data < (resin_threshold * 255)] = 0
                cur_data[cur_data != 0] = 1
                if cv_resin_data is None:
                    cv_resin_data = cur_data
                else:
                    cv_resin_data = np.bitwise_or(cur_data, cv_resin_data)
            cv_resin_data = cv_resin_data * 255

            cv_resin_data_ups = np.array(
                Image.fromarray(cv_resin_data).resize(
                    tuple(resin_scale_factor * v for v in cv_resin_data.shape),
                    resample=Image.NEAREST,
                )
            ).astype(np.uint8)
            spoof_sample["src_plastic"] = cv_resin_data_ups
            spoof_sample["tgt_plastic"] = cv_resin_data_ups
        norm_transform = generate_transform(
            {
                img_mip: [
                    {"type": "preprocess"},
                    {
                        "type": "sergiynorm",
                        "mask_plastic": True,
                        "low_threshold": -0.485,
                        # "high_threshold": 0.35, #BASILLLL
                        "high_threshold": 0.22,  # MINNIE
                        "filter_black": True,
                        "bad_fill": -20.0,
                    },
                ]
            },
            img_mip,
            img_mip,
        )
        norm_sample = apply_transform(spoof_sample, norm_transform)
        processed_patch = norm_sample["src"].squeeze()

        cv_processed_data = get_np(processed_patch.unsqueeze(2).unsqueeze(2)).astype(
            np.float32
        )

        print(z, np.mean(cv_img_data), np.mean(cv_processed_data))

        img_dst_cv[
            cv_xy_start[0] : cv_xy_end[0], cv_xy_start[1] : cv_xy_end[1], z
        ] = cv_processed_data
        e = time.time()
        print(e - s, " sec")


def work(tq, ls):
    tq.poll(lease_seconds=ls)


if __name__ == "__main__":
    with TaskQueue(sys.argv[2]) as tq:
        if sys.argv[1] == "worker":
            work(tq, int(sys.argv[3]))
        elif sys.argv[1] == "master":
            # TODO: proper argument parsing
            start = int(sys.argv[5])
            end = int(sys.argv[6])
            for i in range(start, end):
                tq.insert(
                    NormalizeTask(
                        start_section=i, end_section=1 + i, img_src_cv_path=sys.argv[3], resin_text_path=sys.argv[4]
                    )
                )
