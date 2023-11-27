
Install
============

Dependencies
------------
The AcouPipe package depends on the following packages:

* Acoular_
* Ray_
* h5py_
* tqdm_
* pooch_
* Tensorflow_ (optional)
* Pandas_ (optional)

How to install AcouPipe locally
--------------------------------

Download or clone the AcouPipe repository and enter the directory, e.g. change ``</path/to/dir>`` to the desired path and execute:

.. code-block:: bash

   DIR=</path/to/dir> && git clone git@github.com:adku1173/acoupipe.git $DIR && cd $DIR && unset DIR


Next, install module with pip. This will install all necessary `dependencies`_.

Depending on the purpose of the installation, different options are available. A minimal installation can be done with:

.. code-block:: bash

   pip install .

If you would like to include optional dependencies, install with:

.. code-block:: bash

   pip install ".[full]"


If you are interested in the development version, install with:

.. code-block:: bash

   pip install ".[dev]"


Using a pre-build Docker image
------------------------------

If you are familiar with Docker_, the easiest way to use AcouPipe is by using an existing Docker image from DockerHub_. There are several images available, each tagged with the version of AcouPipe that is installed. The latest version is tagged as ``latest``.

The following images are available:

* ``adku1173/acoupipe:latest`` 
* ``adku1173/acoupipe:latest-full`` 
* ``adku1173/acoupipe:latest-dev`` 
* ``adku1173/acoupipe:latest-jupyter`` 

If  Docker_ is allready installed, simply pull the latest image with the command

.. code-block:: bash

    docker pull adku1173/acoupipe:latest


