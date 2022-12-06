Simulation with Singularity on a High-Performance Cluster (HPC)
---------------------------------------------------------------

If you plan to simulate the data by means of multiple machines (e.g. on a high-performance cluster (HPC))
you can use the `Ray Cluster`_ interface.

The following code snippet gives an example of a job script that can
be scheduled with the SLURM_ job manager and by using a Singularity_ image. 

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=acoupipe_dataset
    #SBATCH --cpus-per-task=16 
    #SBATCH --nodes=4
    #SBATCH --tasks-per-node=1 # Give all resources to a single Ray task, ray can manage the resources internally
    #SBATCH --output=acoupipe_dataset.stdout

    SPLIT="validation"
    SIZE=10000
    DIRPATH=<path-to-the-acoupipe-dataset-directory>
    IMGNAME=<name-of-the-singularity-image> 

    let "worker_num=(${SLURM_NTASKS} - 1)" ### The variable $SLURM_NTASKS gives the total number of cores requested in a job. (tasks-per-node * nodes)-1 
    echo "Number of workers" $worker_num

    # Define the total number of CPU cores available to ray
    let "total_cores=${worker_num} * ${SLURM_CPUS_PER_TASK}"

    suffix='6379'
    ip_head=`hostname`:$suffix
    export ip_head # Exporting for latter access by trainer.py
    echo $ip_head

    # Start the ray head node on the node that executes this script by specifying --nodes=1 and --nodelist=`hostname`
    # We are using 1 task on this node and 5 CPUs (Threads). Have the dashboard listen to 0.0.0.0 to bind it to all
    # network interfaces. This allows to access the dashboard via port-forwarding.
    srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=`hostname` singularity exec -B $DIRPATH $IMGNAME ray start --head --block --dashboard-host 0.0.0.0 --port=6379 --num-cpus ${SLURM_CPUS_PER_TASK} &
    sleep 10

    # Now we execute worker_num worker nodes on all nodes in the allocation except hostname by
    # specifying --nodes=${worker_num} and --exclude=`hostname`. Use 1 task per node, so worker_num tasks in total
    # (--ntasks=${worker_num}) and 5 CPUs per task (--cps-per-task=${SLURM_CPUS_PER_TASK}).
    srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=`hostname` singularity exec -B $DIRPATH $IMGNAME ray start --address $ip_head --block --num-cpus ${SLURM_CPUS_PER_TASK} &
    sleep 10

    singularity exec -B $DIRPATH $IMGNAME python -u $DIRPATH/main.py --head=${ip_head} --tasks=${total_cores} --split=${SPLIT} --size=${SIZE}
