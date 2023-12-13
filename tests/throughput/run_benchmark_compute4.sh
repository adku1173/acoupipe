#!/bin/bash

DOCKER_NAME="acoupipe_benchmark_throughput"
#export MPLCONFIGDIR='/tmp/mpl_config' 

docker pull adku1173/acoupipe:dev-dev

docker run -it -e NUMBA_CACHE_DIR='/tmp/numba_cache' -e TF_CPP_MIN_LOG_LEVEL='3' \
 -e NUMBA_NUM_THREADS='1' -e OMP_NUM_THREADS='1' -e MKL_NUM_THREADS='1' \
 -e OPENBLAS_NUM_THREADS='1' -e BLIS_NUM_THREADS='1' \
 --user "$(id -u)":"$(id -g)" --group-add 1007 -v $(pwd):"/workspace" \
 --name=$DOCKER_NAME --shm-size=30.00gb  adku1173/acoupipe:dev-dev bash -c 'python -u /workspace/main.py --task_list 1 2 4 8 16 32 --size 100 --datasets DatasetMIRACLE DatasetSynthetic --srirdir="/workspace"'

