#!/bin/bash
docker build -t seunglab/ncc_inference:fly_x3 .
docker push seunglab/ncc_inference:fly_x3
