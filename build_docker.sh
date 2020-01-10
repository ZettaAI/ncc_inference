#!/bin/bash
docker build -t seunglab/ncc_inference:fly .
docker push seunglab/ncc_inference:fly
