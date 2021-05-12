#!/bin/bash

TSAMPLES=500000
VSAMPLES=10000
VSTART=51
F=13
TASKS=2
FEATURE="csmtriu sourcemap csm"
FILEFORMAT="h5"

python main.py --file_format=$FILEFORMAT --tasks=$TASKS --datasets validation --tsamples=$TSAMPLES --vsamples=$VSAMPLES --vstart=$VSTART --feature $FEATURE --freq_index=$F
