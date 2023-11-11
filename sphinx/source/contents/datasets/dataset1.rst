
DatasetSynthetic1
=================

.. _fig-msm-layout: 

.. figure:: ../../_static/msm_layout.png
    :width: 750
    :align: center

.. toctree::
    :maxdepth: 1
    :caption: Class Documentation:

    ../../autoapi/acoupipe/datasets/synthetic/index


`DatasetSynthetic1` is a purely synthetic microphone array source case generator following the illustrated default virtual simulation setup with a 64 channel microphone array and a planar observation area. See `Environmental Characteristics`_ for details. 

.. _Environmental Characteristics:

.. table:: Environmental Characteristics

    ===================== ========================================  
    Environment           Anechoic, Resting, Homogeneous Fluid
    Speed of sound        343 m/s
    Microphone Array      Vogel's spiral, :math:`M=64`, Aperture Size 1 m
    Observation Area      x,y in [-0.5,0.5], z=0.5
    Source Type           Monopole 
    Source Signals        Uncorrelated White Noise (:math:`T=5\,s`)
    ===================== ========================================


The underlying default FFT parameters are:

.. table:: FFT Parameters

    ===================== ========================================  
    Sampling Rate         He = 40, fs=13720 Hz 
    Block size            128 Samples
    Block overlap         50 %
    Windowing             von Hann / Hanning
    ===================== ========================================


Randomized Properties
---------------------

Several properties of the dataset are randomized for each source case when generating the data. Table `Randomized properties`_ lists the randomized properties and their respective distributions, which are are closely related to the work of Herold and Sarradj :cite:`Herold2017`. As such, the the microphone positions are spatially disturbed to account for uncertainties in the microphone placement. The number of sources is randomly chosen from a Poisson distribution with a mean of 3. Note that the maximum number of occuring sources is limited to 10 by default.
The source positions are randomly chosen from a bivariate normal distribution with a standard deviation of 0.1688 m. The source strength is randomly chosen from a Rayleigh distribution with a standard deviation of 5 :math:`{Pa}^2` at the reference position. Uncorrelated white noise is added to the microphone signals with a relative variance randomly chosen from a uniform distribution between :math:`10^{-6}` and 0.1 :math:`{Pa}^2`.

.. _Randomized properties:

.. table:: Randomized properties

    ==================================================================   ===================================================  
    Sensor Position Deviation [m]                                        Bivariate normal distributed (:math:`\sigma` = 0.001)
    No. of Sources                                                       Poisson distributed (:math:`\lambda`=3)
    Source Positions                                                     Normal distributed (:math:`\sigma` = 0.1688) 
    Source Strength (:math:`{Pa}^2` at reference position)               Rayleigh distributed (:math:`\sigma_{R}`=5)
    Noise Variance (:math:`{Pa}^2`)                                      Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
    ==================================================================   ===================================================


