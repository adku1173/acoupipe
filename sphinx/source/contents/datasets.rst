.. _data:

Datasets
========
This section gives an overview over the available datasets in AcouPipe.
Currently, AcouPipe provides two default synthetic datasets of stationary noise sources:

.. toctree::
    :maxdepth: 1

    datasets/dataset1
    datasets/dataset2

Dataset1 relies on simulated time data from which the features are extracted has beed used in the following 
publications: :cite:`Kujawski2019`, :cite:`Kujawski2022`, :cite:`Feng2022`.
Dataset2 relies on source covariances sampled from a wishart distribution with non-flat randomly sampled source spectra. 

--------

Examples 
--------

.. toctree::
    :maxdepth: 1

    datasets/docker
    datasets/singularity
    jupyter/generate
    jupyter/load_h5
    jupyter/load_tfrecord
    jupyter/evaluate




