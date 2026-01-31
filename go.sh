#!/bin/bash

CUDA_VISIBLE_DEVICES=${1:-0} # default to use GPU 0

export CUDA_VISIBLE_DEVICES
python main.py --overwrite_args