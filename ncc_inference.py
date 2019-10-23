import scipy
import json
import torch
import numpy as np
import time
import os
import sys
import copy

from copy import deepcopy

import cloudvolume as cv

from pdb import set_trace as st
from helpers import normalize, create_model, get_np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

start_section = int(sys.argv[1])
end_section = int(sys.argv[2])

img_in_out_mip = [(6, 6), (6, 7), (7,8)]

path_template ='gs://microns-seunglab/minnie_v4/processed/'

for  img_in_mip, img_out_mip in img_in_out_mip:
    pyramid_name = "ncc_m{}".format(img_out_mip)
    if img_out_mip == 6:
        cv_src_path = path_template + 'm6_normalized_test'
        cv_dst_path = path_template + 'ncc/ncc_m{}_test'.format(img_out_mip)
    elif img_out_mip in [7, 8]:
        cv_src_path = path_template + 'ncc/ncc_m{}_test'.format(img_in_mip)
        cv_dst_path = path_template + 'ncc/ncc_m{}_test'.format(img_out_mip)
    else:
        raise Exception("Unkown mip")

    cv_src = cv.CloudVolume(cv_src_path, mip=img_in_mip, fill_missing=True, bounded=False, progress=False)
    cv_dst = cv.CloudVolume(cv_dst_path, mip=img_out_mip,
                            fill_missing=True, bounded=False,
                            progress=False, parallel=5,
                            info=deepcopy(cv_src.info), non_aligned_writes=True)
    cv_dst.info['data_type'] = 'float32'
    cv_dst.commit_info()

    cv_xy_start = [0, 0]

    crop = 256
    if img_in_mip == 6:
        cv_xy_start = [256*0, 1024*0]
        cv_xy_end = [8096, 8096]#[1024 * 8 - 256*0, 1024 * 8 - 256*0]
        patch_size = 8096 // 4
    elif img_in_mip == 7:
        cv_xy_start = [256*0, 1024*0]
        cv_xy_end = [4048, 4048]#[1024 * 8 - 256*0, 1024 * 8 - 256*0]
        patch_size = 4048// 2
    elif img_in_mip == 8:
        cv_xy_end = [2024, 2048]#[1024 * 8 - 256*0, 1024 * 8 - 256*0]
        patch_size = 2024

    global_start = 0
    scale_factor = 2**(img_out_mip - img_in_mip)

    encoder = create_model("model",
                 checkpoint_folder="./models/{}".format(pyramid_name))

    for z in range(start_section, end_section):
        print ("MIP {} Section {}".format(img_out_mip, z))
        s = time.time()

        cv_src_data = cv_src[cv_xy_start[0]:cv_xy_end[0], cv_xy_start[1]:cv_xy_end[1], z].squeeze()
        src_data = torch.cuda.FloatTensor(cv_src_data)
        src_data = src_data.unsqueeze(0)

        in_shape = src_data.shape

        dst = torch.zeros((1, in_shape[-2]//scale_factor, in_shape[-1]//scale_factor),
                                                                    device=src_data.device)

        for i in range(0, src_data.shape[-2]//patch_size):
            for j in range(0, src_data.shape[-1]//patch_size):

                x = [global_start + i*patch_size, global_start + (i + 1) * patch_size]
                y = [global_start + j*patch_size, global_start + (j + 1) * patch_size]
                x_padded = copy.copy(x)
                y_padded = copy.copy(y)
                if i != 0:
                    x_padded[0] = x[0] - crop
                if i != src_data.shape[-2]//patch_size - 1:
                    x_padded[1] = x[1] + crop
                if j != 0:
                    y_padded[0] = y[0] - crop
                if j != src_data.shape[-1]//patch_size - 1:
                    y_padded[1] = y[1] + crop

                patch = src_data[..., x_padded[0]:x_padded[1], y_padded[0]:y_padded[1]].squeeze()
                with torch.no_grad():
                    processed_patch = encoder(patch.unsqueeze(0).unsqueeze(0)).squeeze()
                if i != 0:
                    processed_patch = processed_patch[crop//scale_factor:, :]
                if i != src_data.shape[-2]//patch_size - 1:
                    processed_patch = processed_patch[:-crop//scale_factor, :]
                if j != 0:
                    processed_patch = processed_patch[:, crop//scale_factor:]
                if j != src_data.shape[-1]//patch_size - 1:
                    processed_patch = processed_patch[:, :-crop//scale_factor]
                dst[..., x[0]//scale_factor:x[1]//scale_factor,
                        y[0]//scale_factor:y[1]//scale_factor] = processed_patch
                if torch.any(processed_patch != processed_patch):
                    st()

        with torch.no_grad():
            if scale_factor == 2:
                black_mask = src_data != 0
                black_frac = float(torch.sum(black_mask == False)) / float(torch.sum(src_data > -10000))
                black_mask = torch.nn.MaxPool2d(2)(black_mask.unsqueeze(0).float()) != 0
                black_mask = black_mask.squeeze(0)
            elif scale_factor == 4:
                black_mask = src_data != 0
                black_frac = float(torch.sum(black_mask == False)) / float(torch.sum(src_data > -10000))
                black_mask = torch.nn.MaxPool2d(2)(black_mask.unsqueeze(0).float()) != 0
                black_mask = black_mask.squeeze(0)
                black_mask = torch.nn.MaxPool2d(2)(black_mask.unsqueeze(0).float()) != 0
                black_mask = black_mask.squeeze(0)
            elif scale_factor == 1:
                black_mask = (src_data > -10)  * (src_data != 0)
                black_frac = float(torch.sum(black_mask == False)) / float(torch.sum(src_data > -10000))
            else:
                raise Exception("Unimplemented")

            if torch.any(dst != dst):
                st()
            dst_norm = normalize(dst, mask=black_mask, mask_fill=0)
            if torch.any(dst_norm != dst_norm):
                st()
        cv_data = get_np(dst_norm.squeeze().unsqueeze(2).unsqueeze(2)).astype(np.float32)

        cv_dst[cv_xy_start[0]//scale_factor:cv_xy_end[0]//scale_factor,
                cv_xy_start[1]//scale_factor:cv_xy_end[1]//scale_factor, z] = cv_data

        e = time.time()
        print (e - s, " sec")
