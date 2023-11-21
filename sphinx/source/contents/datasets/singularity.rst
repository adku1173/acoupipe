Simulation with Singularity on a High-Performance Cluster (HPC)
---------------------------------------------------------------

If you plan to simulate the data by means of multiple machines (e.g. on a high-performance cluster (HPC))
you can use the `Ray Cluster`_ interface.

If your HPC supports Singularity_ you can easily create a singularity image named `acoupipe.sif` by bootstrapping from an existing docker image from DockerHub_. E.g.:

.. code-block:: bash

    singularity build acoupipe.sif docker://adku1173/acoupipe:latest-full

The following code snippet gives an example of a job script that can
be scheduled with the SLURM_ job manager and by using a Singularity_ image. 

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=<your job name>
    #SBATCH --cpus-per-task=16
    #SBATCH --nodes=2 # two computational nodes
    #SBATCH --tasks-per-node=1 # Give all resources to a single Ray task, ray can manage the resources internally

    # limit the number of threads. Ideally, this should be set
    # such that the number of threads is equal to the total number of threads devided by the 
    # number of parallel executing AcouPipe tasks
    export NUMBA_NUM_THREADS=1 
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export TF_CPP_MIN_LOG_LEVEL=3 # silence TensorFlow if used

    IMGNAME=<path to your image>/acoupipe.sif

    let "worker_num=(${SLURM_NTASKS}-1)" # The variable $SLURM_NTASKS gives the total number of cores requested in a job. (tasks-per-node * nodes)
    let "cpu_num=(${SLURM_NTASKS})"
    echo "Number of workers" $worker_num

    # Define the total number of CPU cores available to ray
    let "total_cores=${cpu_num} * ${SLURM_CPUS_PER_TASK}"

    port=6379
    hname=$(hostname -I | awk '{print $1}') # first IP address given
    ip_head=$hname:$port
    export ip_head # Exporting for latter access by trainer.py
    echo $ip_head

    # Start the ray head node on the node that executes this script by specifying --nodes=1 and --nodelist=`hostname`
    # We are using 1 task on this node and 5 CPUs (Threads). Have the dashboard listen to 0.0.0.0 to bind it to all
    # network interfaces. This allows to access the dashboard via port-forwarding.
    srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=`hostname` singularity exec -B $(pwd) $IMGNAME ray start --head --block --port=$port --num-cpus ${SLURM_CPUS_PER_TASK} &
    sleep 10

    # Now we execute worker_num worker nodes on all nodes in the allocation except hostname by
    # specifying --nodes=${worker_num} and --exclude=`hostname`. Use 1 task per node, so worker_num tasks in total
    # (--ntasks=${worker_num}) and 5 CPUs per task (--cps-per-task=${SLURM_CPUS_PER_TASK}).
    srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=`hostname` singularity exec -B $(pwd) $IMGNAME ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
    sleep 10

    singularity exec -B $(pwd) $IMGNAME python -u $(pwd)/main.py --head=${ip_head} --task_list ${total_cores}
