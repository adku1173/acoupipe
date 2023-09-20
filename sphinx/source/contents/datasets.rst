.. _data:


Datasets
========

Currently, AcouPipe provides two default synthetic datasets of stationary noise sources with similar virtual measurement setup:

.. figure:: ../../_static/msm_layout.png
    :width: 750

**Dataset1** relies on simulated time data from which the features are extracted and has been used in the following 
publications: :cite:`Kujawski2019`, :cite:`Kujawski2022`, :cite:`Feng2022`.

**Dataset2** relies on analytically or sampled auto- and cross-power spectra of the microphones.  

.. toctree::
    :maxdepth: 1

    datasets/theory
    datasets/dataset1
    datasets/dataset2
    datasets/working

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




