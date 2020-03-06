#!/bin/bash
docker build -t seunglab/ncc_inference:missd_x3 .
docker push seunglab/ncc_inference:missd_x3
