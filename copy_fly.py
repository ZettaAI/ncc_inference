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
from taskqueue import RegisteredTask, TaskQueue

import argparse
from multiprocessing import Process

class FlyCopyTask(RegisteredTask):
    def __init__(self, cv_src_path, cv_dst_path, mip, threads, src_chunk_coords,
                 tgt_chunk_coords):
        super(FlyCopyTask, self).__init__(cv_src_path, cv_dst_path, mip, threads,
                                          src_chunk_coords, tgt_chunk_coords)
        self.cv_src_path = cv_src_path
        self.cv_dst_path = cv_dst_path
        self.mip = mip
        self.threads = threads
        self.src_chunk_coords = src_chunk_coords
        self.tgt_chunk_coord = tgt_chunk_coords

    def execute(self):
        s = time.time()
        cv_src = cv.CloudVolume(self.cv_src_path, mip=self.mip, fill_missing=True,
                                bounded=False, progress=False, parallel=self.threads)

        cv_dst = cv.CloudVolume(self.cv_dst_path, mip=self.mip, fill_missing=True,
                                bounded=False, progress=False, parallel=self.threads)
        e = time.time()
        print ("CV creation time: {}".format(e - s))

        s = time.time()
        src_chunk_coords = self.src_chunk_coords
        tgt_chunk_coords = self.tgt_chunk_coords

        cv_src_data = cv_src[src_chunk_coords[0][0]:src_chunk_coords[0][1],
                             src_chunk_coords[1][0]:src_chunk_coords[1][1],
                             src_chunk_coords[2][0]:src_chunk_coords[2][1]]
        cv_dst[tgt_chunk_coords[0][0]:tgt_chunk_coords[0][1],
               tgt_chunk_coords[1][0]:tgt_chunk_coords[1][1],
               tgt_chunk_coords[2][0]:tgt_chunk_coords[2][1]] = cv_src_data

        e = time.time()
        print ("Copy time: {}".format(e - s))

def work(tq, lease_seconds=30):
    tq.poll(lease_seconds=int(lease_seconds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Coppy da fly yo')
    parser.add_argument('--queue', type=str, default=None)
    parser.add_argument('--mode', type=str, default='master')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--worker_proc_count', type=int, default=4)
    parser.add_argument('--lease_seconds', type=int, default=32)
    args = parser.parse_args()
    print ('{} mode activated'.format(args.mode))

    if args.mode == 'master':
        copy_info = True
        cv_src_path ='gs://fafb_v15_montages/FAFB_montage/v15_montage_20190901_rigid_split/1024x1024'
        cv_dst_path = 'gs://fafb_v15_montages/sergiy_playground/test1'
        mip = 0

        print ("Kreating source cloud volume")
        cv_src = cv.CloudVolume(cv_src_path, mip=mip, fill_missing=True, bounded=False,
                                progress=False)
        print ("Kreating dst cloud volume")
        if copy_info:
            cv_dst = cv.CloudVolume(cv_dst_path, mip=mip,
                                    fill_missing=True, bounded=False,
                                    progress=False,
                                    info=deepcopy(cv_src.info))
            cv_dst.info['scales'][0]['key'] = '4_4_40'
            cv_dst.info['scales'][0]['resolution'] = [4, 4, 40]

            cv_dst.commit_info()
        else:
            cv_dst = cv.CloudVolume(cv_dst_path, mip=mip,
                                    fill_missing=True, bounded=False,
                                    progress=False)
        src_cv_start = [0, 8192, 300300]
        tgt_cv_start = [0, 0, 3003]

        chunk_mult = 4
        tgt_step_sizes = [1024*chunk_mult, 1024*chunk_mult, 1]
        src_step_sizes = [1024*chunk_mult, 1024*chunk_mult, 100]

        tgt_chunk_sizes = tgt_step_sizes
        src_chunk_sizes = tgt_chunk_sizes

        src_cv_end = [231424, 114688, 310000]
        tgt_cv_end = [tgt_cv_start[i] + \
                      ceil((src_cv_end[i] - src_cv_start[i]) / (src_step_sizes[i] / tgt_step_sizes[i])) \
                      for i in range(3)]

        step_counts = [ceil((src_cv_end[i] - src_cv_start[i]) / src_step_sizes[i]) for i in range(3)]
        iteration_ranges = [list(range(step_counts[i])) for i in range(3)]
        chunk_ids = list(itertools.product(*iteration_ranges))

        print ("Creating tasks")
        tasks = []
        for chunk_id in tqdm(chunk_ids):
            src_chunk_coords = [(src_cv_start[i] + chunk_id[i] * src_step_sizes[i],
                                 src_cv_start[i] + chunk_id[i] * src_step_sizes[i] + src_chunk_sizes[i]) \
                                for i in range(3)]
            tgt_chunk_coords = [(tgt_cv_start[i] + chunk_id[i] * tgt_chunk_sizes[i],
                                 tgt_cv_start[i] + (chunk_id[i] + 1) * tgt_chunk_sizes[i]) \
                                for i in range(3)]
            tasks.append(FlyCopyTask(cv_src_path=cv_src_path, cv_dst_path=cv_dst_path,
                        mip=mip, threads=args.threads, src_chunk_coords=src_chunk_coords,
                        tgt_chunk_coords=tgt_chunk_coords))

    if args.queue is None:
        if mode == 'master':
            print ("Executing tasks")
            #local version
            for task in tqdm(tasks):
                task.execute()
        else:
            raise Exception ('Local worker mode is not supported')
    else:
        #distributed version
        with TaskQueue(args.queue) as tq:
            if args.mode == "worker":
                print ("Working")
                worker_ps = []

                for _ in range(args.worker_proc_count):
                    print ('starting another worker')
                    p = Process(target=work, args=(tq, args.lease_seconds))
                    p.start()

            elif args.mode == "master":
                print ("Scheduling Tasks")
                for task in tqdm(tasks):
                    tq.insert(task)
