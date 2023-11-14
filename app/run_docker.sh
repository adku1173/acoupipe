#!/bin/sh

# example docker shell script

SPLIT='training'
#SIZE=<enter desired size of the dataset>
SIZE=1000
#NTASKS=<enter the number of parallel tasks> # should match the number of CPUs on the host
NTASKS=2

docker run -p 8265:8265 -e NUMBA_NUM_THREADS=1 -it --user "$(id -u)":"$(id -g)" -v $(pwd):/app adku1173/acoupipe:dev-full python /app/main.py --tasks=$NTASKS --size=$SIZE --split=$SPLIT --features 'csm'

