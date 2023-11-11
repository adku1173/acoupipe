#!/bin/bash

#!/bin/bash

DOCKER_NAME="acoupipe_benchmark_throughput"
MNT_PATH="/home/kujawski/docker/git_pkg/acoupipe/tests/throughput"

docker run --user "$(id -u)":"$(id -g)" --group-add 1007 -v $MNT_PATH:"/workspace" --name=$DOCKER_NAME -it --rm --shm-size=10.24gb d421120aba77 \
bash -c "export NUMBA_CACHE_DIR='/tmp/numba_cache' && export MPLCONFIGDIR='/tmp/mpl_config' && export TF_CPP_MIN_LOG_LEVEL='2' &&
export NUMBA_NUM_THREADS='1' && export NUMBA_THREADING_LAYER='tbb' && export OMP_NUM_THREADS='1' && python -u /workspace/main.py --task_list 1 2 4 8 16 32 --size 100"
