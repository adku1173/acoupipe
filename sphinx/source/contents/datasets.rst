.. _data:

Datasets
========
This section gives an overview over the available datasets in AcouPipe.
Currently, AcouPipe provides two default synthetic datasets of stationary noise sources:

* :doc:`datasets/dataset1` 
* :doc:`datasets/dataset2`

Dataset1 relies on simulated time data from which the features are extracted has beed used in the following publications: [Kuj19]_, [Kuj22]_, [Fen22]_.
Dataset2 relies on source covariances sampled from a wishart distribution with non-flat randomly sampled source spectra. 

--------

.. toctree::
    :maxdepth: 1

    jupyter/generate
    datasets/docker
    datasets/singularity
    jupyter/load_h5
    jupyter/load_tfrecord



