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


def normalize_cv(src_cv, x, y, z, mean_cv, var_cv, fill_value=0,
                 mask_cv=None, th_high=None, th_low=None):
    data = np.array(src_cv[x[0]:x[1], y[0]:y[1], z]).squeeze()
    if mask_cv is None:
        mask = np.zeros_like(data) > 0
    else:
        mask = np.array(mask_cv[x[0]:x[1], y[0]:y[1], z]).squeeze() > 0

    if th_low is not None:
        mask[data < th_low] = 1
    if th_high is not None:
        mask[data > th_high] = 1
    mask[data == 0] = 1
    mask[data == -20] = 1

    mean = np.array(mean_cv[0, 0, z]).squeeze()
    var = np.array(var_cv[0, 0, z]).squeeze()
    result = deepcopy(data).astype(np.float32)
    result[mask == False] -= mean
    result[mask == False] /= np.sqrt(var)
    result[mask] = fill_value

    return result

class NormalizeTask(RegisteredTask):
    def __init__(self, start_section, end_section, img_src_cv_path):
        #resin_cv_path = os.path.join(img_src_cv_path, "resin_mask")
        resin_cv_path = None
        img_dst_cv_path = os.path.join(img_src_cv_path, "m4_normalized")
        super(NormalizeTask, self).__init__(start_section, end_section, img_src_cv_path)

        # attributes passed to super().__init__ are automatically assigned
        # use this space to perform additional processing such as:
        self.start_section = int(start_section)
        self.end_section = int(end_section)

        self.img_dst_cv_path = img_dst_cv_path
        self.img_src_cv_path = img_src_cv_path
        self.resin_cv_path = resin_cv_path

    def execute(self):
        if self.start_section and self.end_section:
            normalize_section_range(
                self.start_section,
                self.end_section,
                self.img_dst_cv_path,
                self.img_src_cv_path,
                self.resin_cv_path,
            )
        else:
            print(self)


def normalize_section_range(
    start_section, end_section, img_dst_cv_path, img_src_cv_path, resin_cv_path
):
    resin_mip = 8
    img_mip = 4

    img_src_cv = cv.CloudVolume(
        img_src_cv_path,
        mip=img_mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        parallel=1,
    )

    img_dst_cv = cv.CloudVolume(
        img_dst_cv_path,
        mip=img_mip,
        fill_missing=True,
        progress=False,
        bounded=False,
        parallel=1,
        info=deepcopy(img_src_cv.info),
        non_aligned_writes=False,
        autocrop=True,
        delete_black_uploads=True,
    )

    img_dst_cv.info["data_type"] = "float32"
    img_dst_cv.commit_info()

    if resin_cv_path is not None:
        resin_cv = cv.CloudVolume(
            resin_cv_path,
            mip=resin_mip,
            fill_missing=True,
            progress=False,
            bounded=False,
            parallel=1,
        )

    resin_scale_factor = 2 ** (resin_mip - img_mip)

    #cv_xy_start = [224256//16, 158720//16]
    #MINNIE
    #cv_xy_end = [8096, 8096]
    #FLY
    cv_xy_start = [13248, 9152]
    cv_xy_end = [16320, 13248]
    spoof_sample = {"src": None, "tgt": None}

    for z in range(start_section, end_section):
        print(z)
        s = time.time()
        cv_img_data = img_src_cv[
            cv_xy_start[0] : cv_xy_end[0], cv_xy_start[1] : cv_xy_end[1], z
        ].squeeze()

        spoof_sample["src"] = cv_img_data
        spoof_sample["tgt"] = cv_img_data

        if resin_cv_path is not None:
            cv_resin_data = resin_cv[
                cv_xy_start[0]
                // resin_scale_factor : cv_xy_end[0]
                // resin_scale_factor,
                cv_xy_start[1]
                // resin_scale_factor : cv_xy_end[1]
                // resin_scale_factor,
                z,
            ].squeeze()

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
                        "mask_plastic": False,
                        "low_threshold": -0.485,
                        # "high_threshold": 0.35, #BASILLLL
                        #"high_threshold": 0.22,  # MINNIE
                        "high_threshold": 0.35,  #FLY
                        "filter_black": True,
                        "bad_fill": -20.0,
                    },
                ]
            },
            img_mip,
            img_mip,
            cpu=True
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


def work(tq):
    tq.poll(lease_seconds=int(20))


if __name__ == "__main__":
    with TaskQueue(sys.argv[2]) as tq:
        if sys.argv[1] == "worker":
            work(tq)
        elif sys.argv[1] == "master":
            # TODO: proper argument parsing
            start = 17005
            end = 18000
            for i in range(start, end):
                tq.insert(
                    NormalizeTask(
                        start_section=i, end_section=1 + i, img_src_cv_path=sys.argv[3]
                    )
                )
