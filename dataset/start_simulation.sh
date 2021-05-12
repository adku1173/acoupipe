#!/bin/bash

TSAMPLES=10
VSAMPLES=50
HE=4
TASKS=2
FEATURE="csmtriu sourcemap csm"
FILEFORMAT="h5"

python main.py --file_format=$FILEFORMAT --tasks=$TASKS --datasets validation --tsamples=$TSAMPLES --vsamples=$VSAMPLES --he=$HE --feature $FEATURE
