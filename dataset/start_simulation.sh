#!/bin/bash

TSAMPLES=10
VSAMPLES=10
HE=4
SETS=training
TASKS=1
FEATURE="csm"
FILEFORMAT="tfrecord"

NUMBA_NUM_THREADS=1 python main.py --log --file_format=$FILEFORMAT --tasks=$TASKS --datasets training validation --tsamples=$TSAMPLES --vsamples=$VSAMPLES --he=$HE --feature=$FEATURE
