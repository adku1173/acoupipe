#!/bin/bash

# 
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

#remove cache data before testing
rm -rf ./cache/*

#build a test suite object which runs the tests in this folder
python -m unittest discover -v -p "test_*.py"

VAL=$?

#remove cache data after testing
rm -rf ./cache/*

exit $VAL


