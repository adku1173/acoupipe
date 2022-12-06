Simulation with Docker
---------------------------------

The easiest way to create the dataset is by using an existing
Docker image from DockerHub_. Simply pull the latest image with the command

.. code-block:: 

    docker pull adku1173/acoupipe:latest

The image contains the simulation source code and an up-to-date version of Acoular_, 
**AcouPipe** and Tensorflow_.
One can run the dataset simulation given by the main.py script from inside the Docker container by typing

.. code-block:: 

    SPLIT="validation"
    SIZE=10000
    NTASKS=<enter the number of parallel tasks> # should match the number of CPUs on the host
    docker run -it --user "$(id -u)":"$(id -g)" -v $PWD:/data/datasets adku1173/acoupipe python main.py --tasks=$NTASKS --size=$SIZE --split=$SPLIT

Note that the current user on the host is specified as the user of the docker environment with the additional argument :code:`--user "$(id -u)":"$(id -g)"`.
It is not recommended to run the container as a root user.
Further, a directory where the dataset files are stored needs to be bound to the container. With the 
:code:`HOSTDIR=$PWD` command, the current working directory on Linux or macOS hosts is bound. 
The simulation can be run on multiple CPU threads in parallel to speed up computations and the exact number of threads can be specified by the 
user with the :code:`--tasks` argument. For demonstration, we calculate the validation split containing 10,000 samples.

After starting the main script, a progress bar should appear that logs the current simulation status:

.. code-block:: 

    1%|█▍                           | 83/10000 [01:04<1:40:35,  1.64it/s]

It is possible to view the CPU usage in a dashboard application served by the Ray_ API. One should find the following output at the beginning 
of the simulation process when running the simulation on multiple CPU threads

.. code-block:: 

    2021-05-14 08:50:16,533	INFO services.py:1267 -- View the Ray dashboard at http://0.0.0.0:8265

It is necessary to forward the corresponding TCP port with :code:`docker run -p 8265:8265 ...` at the start-up of the container to access the server serving the dashboard.
One can open the dashboard by accessing the web address http://0.0.0.0:8265 which should display the following web interface


.. figure:: ../../_static/dashboard.png
    :width: 780


The main.py script has some further command line options that can be used to influence the simulation process:


.. code-block:: bash
    :caption: command line arguments of the main.py script

    usage: main.py [-h] [--dataset {dataset1}] [--split {training,validation,test}] [--features {sourcemap,csmtriu,csm,ref_cleansc} [{sourcemap,csmtriu,csm,ref_cleansc} ...]] [--f F [F ...]] [--num NUM] [--size SIZE] [--startsample STARTSAMPLE] [--name NAME] [--format {tfrecord,h5}] [--tasks TASKS] [--head HEAD] [--config CONFIG] [--log]

    optional arguments:
        -h, --help            show this help message and exit
        --dataset {dataset1}  Which dataset to compute
        --split {training,validation,test}
                                Which dataset split to compute ('training' or 'validation' or 'test')
        --features {sourcemap,csmtriu,csm,ref_cleansc} [{sourcemap,csmtriu,csm,ref_cleansc} ...]
                                Features included in the dataset. Default is the cross-spectral matrix 'csm'
        --f F [F ...]         frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)
        --num NUM             bandwidth of the considered frequencies. Default is single frequency line(s)
        --size SIZE           Total number of samples to simulate
        --startsample STARTSAMPLE
                                Start simulation at a specific sample of the dataset. Default: 1
        --name NAME           filename of simulated data. If 'None' a filename is given and the file is stored under './datasets'
        --format {tfrecord,h5}
                                Desired file format to store the datasets. Defaults to '.h5' format
        --tasks TASKS         Number of asynchronous tasks. Defaults to '1' (non-distributed)
        --head HEAD           IP address of the head node in the ray cluster. Only necessary when running in distributed mode.
        --config CONFIG       Optional config.yml file specifying underlying parameters.
        --log                 Whether to log timing statistics to file. Only for internal use.
