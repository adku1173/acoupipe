
.. _dataset1:

Dataset1
========

.. figure:: ../../_static/msm_layout.png
    :width: 750

.. toctree::
    :maxdepth: 1
    :caption: Class Documentation

    ./../../autoapi/acoupipe/datasets/dataset1/index


Default Characteristics
-------------------------

Environmental Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===================== ========================================  
Environment           Anechoic, Resting, Homogeneous Fluid
Speed of sound        343 m/s
Microphone Array      Vogel's spiral, :math:`M=64`, Aperture Size 1 m
Observation Area      x,y in [-0.5,0.5], z=0.5
Source Type           Monopole 
Sampling Rate         He = 40, fs=13720 Hz 
Source Signals        Uncorrelated White Noise (:math:`T=5\,s`)
No. of Time Samples   68.600 
===================== ========================================

Random Variables
^^^^^^^^^^^^^^^^


==================================================================   ===================================================  
Sensor Position Deviation [m]                                        Normal Distributed (sigma = 0.001)
No. of Sources                                                       Poisson Distributed (lambda=3)
Source Positions                                                     Normal Distributed (sigma = 0.1688) 
Source Strength (:math:`{Pa}^2` at reference microphone)             Rayleigh Distributed (sigma_R=5)
==================================================================   ===================================================

Processing Parameters
^^^^^^^^^^^^^^^^^^^^^

The underlying processing parameters used to calculate the CSM and/or the conventional beamforming map are:

===================== ========================================  
CSM estimation Method Welch's Method
Block size            128 Samples
Block overlap         50 %
Windowing             von Hann / Hanning
Steering vector       Formulation III, see :cite:`Sarradj2012`
===================== ========================================
