import cloudvolume as cv
import json
import torch
import numpy as np

import time
import sys
import os
import scipy
from scipy import misc

from copy import deepcopy

from helpers import get_np
from augment import set_up_transformation, generate_transform, apply_transform

from taskqueue import RegisteredTask, MockTaskQueue, TaskQueue

class NormalizeTask(RegisteredTask):
  def __init__(self, start_section=None, end_section=None,
    img_dst_cv_path = 'gs://seunglab_minnie_phase3/alignment/unaligned/m6_normalized',
    img_src_cv_path = 'gs://seunglab_minnie_phase3/alignment/unaligned',
    resin_cv_path = 'gs://seunglab_minnie_phase3/alignment/unaligned/resin_mask'
                ):
    super(NormalizeTask, self).__init__(start_section, end_section, img_dst_cv_path,
            img_src_cv_path, resin_cv_path)

    # attributes passed to super().__init__ are automatically assigned
    # use this space to perform additional processing such as:
    self.start_section = int(start_section)
    self.end_section = int(end_section)

    self.img_dst_cv_path = img_dst_cv_path
    self.img_src_cv_path = img_src_cv_path
    self.resin_cv_path = resin_cv_path

  def execute(self):
    if self.start_section and self.end_section:
        normalize_section_range(self.start_section, self.end_section,
                self.img_dst_cv_path, self.img_src_cv_path, self.resin_cv_path)
    else:
      print(self)

def normalize_section_range(start_section, end_section,
        img_dst_cv_path, img_src_cv_path, resin_cv_path):
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

def work(tq):
    tq.poll(lease_Seconds=int(300))

if __name__ == "__main__":
    tq = TaskQueue(sys.argv[2])
    #tq = MockTaskQueue()
    if (sys.argv[1] == 'worker'):
        work(tq)
    elif sys.argv[1] == 'master':
        # w000ohhooooo
        start = 14780
        end = 27883
        for i in range(start, end):
            tq.insert(NormalizeTask(i, 1 + i))
        st()

        #work(tq)

     #t = NormalizeTask(15000, 16000)
     #t.execute()
