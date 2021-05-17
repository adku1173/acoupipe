#!/bin/bash

TSAMPLES=500000
VSAMPLES=10000
VSTART=1
F=13
TASKS=4
FEATURE="csmtriu sourcemap"
FILEFORMAT="h5"

python main.py --file_format=$FILEFORMAT --tasks=$TASKS --datasets validation --tsamples=$TSAMPLES --vsamples=$VSAMPLES --vstart=$VSTART --feature $FEATURE --freq_index=$F
