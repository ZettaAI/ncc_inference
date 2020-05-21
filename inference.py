import os
import sys
import time
import json
import copy
import six

import cloudvolume as cv
import numpy as np
from scipy.ndimage.measurements import label
from cloudvolume.lib import Bbox

import torch
from helpers import get_np
from taskqueue import RegisteredTask, TaskQueue
from tissue_mask import tissue_mask_detector_cv
from misalignment_mask import misalignment_mask_detector_cv
from basic_ops import copy_cv
from embed import embed_cv
from stats import get_patch_stats, stats_info, combine_patch_stats
from normalize import normalize_cv

import argparse

def processor_lookup(processor_name):
    lookup_map = {
        'tissue_mask': tissue_mask_detector_cv,
        'misalignment_mask': misalignment_mask_detector_cv,
        'copy': copy_cv,
        'normalize': normalize_cv,
        'embed': embed_cv,
        'stats': None
    }
    return lookup_map[processor_name]

def processor_inference_type_lookup(processor_name):
    lookup_map = {
        'tissue_mask': patchwise_process_3d,
        'misalignment_mask': patchwise_process_3d,
        'copy': patchwise_process_3d,
        'normalize': patchwise_process_3d,
        'embed': patchwise_process_3d,
        'stats': patchwise_stats_3d
    }
    return lookup_map[processor_name]

class ProcessorTask(RegisteredTask):
    def __init__(self, bbox_start, bbox_end, path, dst_path, mip, processor_name,
            processor_params={}, processor_cv_args={}, patch_size=3072, crop=256, suffix=None,
            src_z_step=1, bckp_src_cv_path=None, bckp_cond=None, bckp_src_z_step=None):
        super().__init__(bbox_start, bbox_end, path, dst_path, mip, processor_name,
                processor_params, processor_cv_args, patch_size, crop, suffix, src_z_step,
                bckp_src_cv_path, bckp_cond, bckp_src_z_step)

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
            self.processor_params = json.loads(self.processor_params)
            self.processor_cv_args = json.loads(self.processor_cv_args)
            for k, v in six.iteritems(self.processor_cv_args):
                self.processor_cv_args[k] = cv.CloudVolume(v['path'], mip=v['mip'],
                        bounded=False, progress=False, fill_missing=True)

            inference = processor_inference_type_lookup(self.processor_name)
            inference(self.bbox_start, self.bbox_end, self.path,
                    self.dst_path,
                    self.mip,
                    self.processor_name, self.processor_params,
                    self.processor_cv_args,
                    self.patch_size, self.crop, self.suffix,
                    self.src_z_step,
                    self.bckp_src_cv_path,
                    self.bckp_cond,
                    self.bckp_src_z_step)
        else:
            print(self)

def patchwise_stats_3d(bbox_start, bbox_end, path, dst_path, mip, processor_name,
                       processor_params, processor_cv_args, patch_size, crop, suffix, src_z_step,
                       bckp_src_cv_path=None, bckp_cond=None, bckp_src_z_step=None):
    assert dst_path is None
    assert bckp_src_cv_path is None
    assert crop == 0

    s = time.time()
    processor = processor_lookup(processor_name)
    src_cv_path = path
    src_cv = cv.CloudVolume(
        src_cv_path,
        mip=mip,
        fill_missing=True,
        bounded=False,
        progress=False,
    )

    if suffix is None:
        suffix = ''
    dst_cv_lookup = {}
    for o in processor_params['outputs']:
        dst_cv_path = os.path.join(
            src_cv_path, suffix, o
        )
        dst_cv = cv.CloudVolume(
            dst_cv_path,
            mip=0,
            fill_missing=True,
            bounded=False,
            info=stats_info,
            progress=False,
        )
        dst_cv_lookup[o] = dst_cv
        dst_cv.commit_info()

    bbox = Bbox([i // 2**mip for i in bbox_start[0:2]] + bbox_start[2:3],
            [i // 2**mip for i in bbox_end[0:2]] + bbox_end[2:3])
    start_section = bbox_start[2]
    end_section = bbox_end[2]

    xy_start = bbox.minpt[0:2]
    xy_end = bbox.maxpt[0:2]
    patch_stats = []
    for src_z in range(start_section, end_section, src_z_step):
        for i in range(0, (xy_end[0] - xy_start[0] + patch_size - 1)//patch_size):
            for j in range(0, (xy_end[1] - xy_start[1] + patch_size - 1)//patch_size):
                x = [xy_start[0] + i*patch_size, xy_start[0] + (i + 1) * patch_size]
                y = [xy_start[1] + j*patch_size, xy_start[1] + (j + 1) * patch_size]
                patch_stats.append(get_patch_stats(src_cv, x, y, src_z, **(processor_params),
                    **(processor_cv_args)))
                print ("Patch: {}".format(patch_stats))
        final_stats = combine_patch_stats(patch_stats)
        print ("Final: {}".format(final_stats))

        for k, v in six.iteritems(final_stats):
            dst_cv_lookup[k][0, 0, src_z] = v

    e = time.time()
    print(e - s, " sec")


def patchwise_process_3d(bbox_start, bbox_end, path, dst_path, mip, processor_name,
                           processor_params, processor_cv_args, patch_size, crop, suffix, src_z_step,
                           bckp_src_cv_path=None, bckp_cond=None, bckp_src_z_step=None):
    s = time.time()
    processor = processor_lookup(processor_name)
    src_cv_path = path
    if dst_path is None:
        if suffix is None:
            suffix = processor_name
        assert len(suffix) > 0
        dst_cv_path = os.path.join(
            path, suffix
        )
    else:
        dst_cv_path = dst_path

    src_cv = cv.CloudVolume(
        src_cv_path,
        mip=mip,
        fill_missing=True,
        bounded=False,
        progress=False,
    )
    if bckp_src_cv_path is not None:
        bckp_src_cv = cv.CloudVolume(
            bckp_src_cv_path,
            mip=mip,
            fill_missing=True,
            bounded=False,
            progress=False,
        )
    else:
        bckp_src_cv = None

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
    #import pdb; pdb.set_trace()
    if dst_cv.info['scales'][0]['key'] != '4_4_40':
        dst_cv.info['scales'][0]['key'] = '4_4_40'
        dst_cv.info['scales'][0]['resolution'] = [4, 4, 40]
        dst_cv.commit_info()
    if processor_name == 'embed' or processor_name == 'normalize':
        dst_cv.info["data_type"] = "float32"
        dst_cv.commit_info()

    bbox = Bbox([i // 2**mip for i in bbox_start[0:2]] + bbox_start[2:3],
            [i // 2**mip for i in bbox_end[0:2]] + bbox_end[2:3])
    aligned_bbox = bbox.expand_to_chunk_size(src_cv.chunk_size,
                                             src_cv.voxel_offset)
    start_section = bbox_start[2]
    end_section = bbox_end[2]

    xy_start = aligned_bbox.minpt[0:2]
    xy_end = aligned_bbox.maxpt[0:2]
    counter = 0
    print ("Writing to {}".format(dst_cv_path))
    for src_z in range(start_section, end_section, src_z_step):
        dst_z = src_z // src_z_step
        print ("{} -> {}".format(src_z, dst_z))
        for i in range(0, (xy_end[0] - xy_start[0] + patch_size - 1)//patch_size):
            for j in range(0, (xy_end[1] - xy_start[1] + patch_size - 1)//patch_size):
                x = [xy_start[0] + i*patch_size, xy_start[0] + (i + 1) * patch_size]
                y = [xy_start[1] + j*patch_size, xy_start[1] + (j + 1) * patch_size]
                x_pad = copy.deepcopy(x)
                y_pad = copy.deepcopy(y)
                if crop > 0:
                    if i != 0:
                        x_pad[0] -= crop
                    if j != 0:
                        y_pad[0] -= crop
                    if i < ((xy_end[0] - xy_start[0] + patch_size - 1)//patch_size - 1) :
                        x_pad[1] += crop
                    if j < ((xy_end[1] - xy_start[1] + patch_size - 1)//patch_size - 1):
                        y_pad[1] += crop
                #print (x, y)

                #curr_chunk = src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                #prev_chunk = src_cv[x[0]:x[1], y[0]:y[1], z].squeeze()
                result = processor(src_cv, x_pad, y_pad, src_z, **(processor_params),
                        **processor_cv_args)
                if bckp_src_cv is not None and check_condition(result, bckp_cond):
                    bckp_src_z = src_z * bckp_src_z_step // src_z_step
                    result = processor(bckp_src_cv, x_pad, y_pad, bckp_src_z,
                                       **(processor_params), **(processor_cv_args))

                #result = result.astype('float32')
                #print (result.shape)
                if crop > 0:
                    if i != 0:
                        result = result[..., crop:, :]
                    if j != 0:
                        result = result[..., :, crop:]
                    if i < ((xy_end[0] - xy_start[0] + patch_size - 1)//patch_size - 1):
                        result = result[..., :-crop, :]
                    if j < ((xy_end[1] - xy_start[1] + patch_size - 1)//patch_size - 1):
                        result = result[..., :, :-crop]
                counter += 1
                print ("Chunk {} x: {}, y: {}, Avg: {:.2f}".format(counter, x, y, result.mean()))
                dst_cv[x[0]:x[1], y[0]:y[1], dst_z] = result[..., np.newaxis, np.newaxis]

    e = time.time()
    print(e - s, " sec")

def check_condition(x, cond):
    if cond is None or cond == 'False':
        return False
    elif cond == 'True':
        return True
    elif cond == 'zero':
        return np.abs(x).sum() == 0

def work(tq, lease_seconds=200):
    print ("start polling tasks...")
    tq.poll(lease_seconds=int(lease_seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train everything.')

    parser.add_argument('--mode', type=str)
    parser.add_argument('--queue_name', type=str)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--cv_path', type=str, default=None)
    parser.add_argument('--dst_cv_path', type=str, default=None)
    parser.add_argument('--bbox_start', type=str, default=None)
    parser.add_argument('--bbox_end', type=str, default=None)
    parser.add_argument('--mip', type=int, default=4)
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--lease_seconds', type=int, default=260)
    parser.add_argument('--processor_name', '-p', type=str)
    parser.add_argument('--processor_params', type=str, default='{}')
    parser.add_argument('--processor_cv_args',  type=str, default='{}')
    parser.add_argument('--src_z_step', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=3072)
    parser.add_argument('--bckp_src_cv_path', type=str, default=None)
    parser.add_argument('--bckp_cond', type=str, default=None)
    parser.add_argument('--bckp_src_z_step', type=int, default=1)
    parser.add_argument('--z_block', type=int, default=1)

    args = parser.parse_args()
    print ("Hello, world!")

    with TaskQueue(args.queue_name) as tq:
        if args.mode == "worker":
            work(tq, args.lease_seconds)
        elif args.mode == "master":
            bbox_start = [int(i) for i in args.bbox_start.split(',')]
            bbox_end = [int(i) for i in args.bbox_end.split(',')]
            for z in range(bbox_start[-1], bbox_end[-1], args.src_z_step):
                tq.insert(
                    ProcessorTask(
                        mip=args.mip,
                        crop=args.crop,
                        bbox_start=[bbox_start[0], bbox_start[1], z],
                        bbox_end=[bbox_end[0], bbox_end[1], z+1],
                        path =args.cv_path,
                        dst_path=args.dst_cv_path,
                        processor_name=args.processor_name,
                        processor_params=args.processor_params,
                        processor_cv_args=args.processor_cv_args,
                        suffix=args.suffix,
                        src_z_step=args.src_z_step,
                        patch_size=args.patch_size,
                        bckp_src_cv_path=args.bckp_src_cv_path,
                        bckp_cond=args.bckp_cond,
                        bckp_src_z_step=args.bckp_src_z_step
                    )
                )
