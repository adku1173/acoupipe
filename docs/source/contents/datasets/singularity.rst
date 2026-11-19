Simulation with Singularity on a High-Performance Cluster (HPC)
---------------------------------------------------------------

If you plan to simulate the data by means of multiple machines (e.g. on a high-performance cluster (HPC))
you can use the `Ray Cluster`_ interface.

If your HPC supports Singularity_ you can easily create a singularity image named `acoupipe.sif` by bootstrapping from an existing docker image from DockerHub_. E.g.:

.. code-block:: bash

    singularity build acoupipe.sif docker://adku1173/acoupipe:latest-full

The image contains the ``app/main.py`` script of the AcouPipe repository (see :doc:`docker`) under ``/app/main.py``.
It can be executed directly from the image. Since the image is read-only, the output file has to be
specified with the ``--name`` argument in this case.

Alternatively, e.g. to modify the script, copy it from the image (or from the ``app`` directory of the AcouPipe repository)
to the working directory and replace ``/app/main.py`` by ``$(pwd)/main.py`` in the job script below.
Without ``--name``, the dataset is then stored in ``$(pwd)/datasets``.

.. code-block:: bash

    singularity exec acoupipe.sif cp /app/main.py .

The following code snippet gives an example of a job script that can
be scheduled with the SLURM_ job manager and by using a Singularity_ image.
It starts one Ray process per node and simulates the dataset on all allocated CPU cores.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=<your job name>
    #SBATCH --nodes=2 # two computational nodes
    #SBATCH --ntasks-per-node=1 # one Ray process per node, Ray manages the resources internally
    #SBATCH --cpus-per-task=16 # CPU cores per node available to Ray
    #SBATCH --time=01:00:00

    # limit the number of threads per Ray worker to avoid oversubscription
    export NUMBA_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export BLIS_NUM_THREADS=1

    IMGNAME=<path to your image>/acoupipe.sif
    SIZE=<number of samples>

    worker_num=$((SLURM_JOB_NUM_NODES - 1)) # all nodes except the head node
    total_cores=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK)) # CPU cores available to Ray

    head_node=$(hostname)
    port=6379
    ip_head=$(hostname -I | awk '{print $1}'):$port # first IP address of the head node
    echo "Ray head: $ip_head, worker nodes: $worker_num, total cores: $total_cores"

    # Compile and cache the Numba functions once on every node. Otherwise, the Ray workers
    # on a node compile them concurrently, which can crash the workers.
    srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks=${SLURM_JOB_NUM_NODES} --cpus-per-task=1 singularity exec -B $(pwd) $IMGNAME python -c "import acoupipe.datasets.synthetic"

    # start the Ray head node on the node that executes this script
    srun --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --nodelist=${head_node} singularity exec -B $(pwd) $IMGNAME ray start --head --block --port=$port --num-cpus=${SLURM_CPUS_PER_TASK} &
    sleep 10

    # start one Ray worker on each of the remaining nodes
    srun --nodes=${worker_num} --ntasks=${worker_num} --cpus-per-task=${SLURM_CPUS_PER_TASK} --exclude=${head_node} singularity exec -B $(pwd) $IMGNAME ray start --address=$ip_head --block --num-cpus=${SLURM_CPUS_PER_TASK} &
    sleep 10

    # simulate the dataset (see the docker section for further options of main.py)
    singularity exec -B $(pwd) $IMGNAME python -u /app/main.py --head=$ip_head --tasks=$total_cores --size=$SIZE --features csm --name=$(pwd)/dataset.h5

The Ray processes are terminated by SLURM_ when the job script finishes.
