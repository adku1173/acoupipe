#!/bin/bash

export NUMBA_NUM_THREADS=1
export NUMBA_THREADING_LAYER=tbb
export OMP_NUM_THREADS=1

IMGNAME=../environment/acoupipe_test.sif

#module load singularity

tasks=("1" "2" "4" "8" "16")
for dataset in "DatasetSynthetic1" "DatasetMIRACLE"; do
    for task in "${tasks[@]}"; do
        echo "Running $dataset, $task tasks"
        srun --job-name="D${task}" --nodes=1 --ntasks-per-node=1 --exclusive --output="${dataset}_${task}" singularity exec -B $(pwd) $IMGNAME python -u $(pwd)/main.py --datasets $dataset --task_list $task --size 100 &
        #srun --partition=TestAndBuild --time=5:00 --job-name="D${task}" --nodes=1 --ntasks-per-node=1 --exclusive --output="${dataset}_${task}" singularity exec -B $(pwd) $IMGNAME python -u $(pwd)/main.py --datasets $dataset --task_list $task --size 100 &
    done
done

