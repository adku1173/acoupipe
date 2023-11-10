#!/bin/bash

export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=tbb
export OMP_NUM_THREADS=1

IMGNAME=../environment/acoupipe_test.sif

singularity exec -B $(pwd) $IMGNAME python -u $(pwd)/main.py --task_list 1 2 4 8 16 32 --size 100 

