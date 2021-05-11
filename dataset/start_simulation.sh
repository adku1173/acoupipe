#!/bin/bash

TSAMPLES=10
VSAMPLES=100
HE=4
TASKS=1
FEATURE="csmtriu sourcemap csm"
FILEFORMAT="tfrecord"

NUMBA_NUM_THREADS=1 python main.py --file_format=$FILEFORMAT --tasks=$TASKS --datasets validation --tsamples=$TSAMPLES --vsamples=$VSAMPLES --he=$HE --feature $FEATURE
