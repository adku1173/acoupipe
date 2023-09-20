
Dataset2
========

.. figure:: ../../_static/msm_layout.png
    :width: 750

Environmental Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

===================== ========================================  
Environment           Anechoic, Resting, Homogeneous Fluid
Speed of sound        343 m/s
Microphone Array      Vogel's spiral, :math:`M=64`, Aperture Size :math:`d_a=1.47\,\mathrm{m}`
Observation Area      x,y in [-0.5:math:`d_a`,0.5:math:`d_a`], z=0.5:math:`d_a`
Source Type           Monopole 
Sampling Rate         He = 219, fs=51200 Hz 
No. of Time Samples   – (no time data available, CSM is calculated directly)
===================== ========================================

Random Variables
^^^^^^^^^^^^^^^^
==================================================================   ===================================================  
Sensor Position Deviation [m]                                        Normal Distributed (sigma = 0.001)
No. of Sources                                                       Poisson Distributed (lambda=3)
Source Positions                                                     Normal Distributed (sigma = :math:`0.1688d_a`) 
Source Strength (:math:`{Pa}^2` at reference microphone)             Rayleigh Distributed (sigma_R=5)
Uncorrelated Noise Variance                                          Uniform (1^-6, 1^-3)
==================================================================   ===================================================

Processing Parameters
^^^^^^^^^^^^^^^^^^^^^

The underlying processing parameters used to calculate the CSM and/or the conventional beamforming map are:

===================== ========================================  
CSM estimation Method Analytic calculation or Sampled according to complex Wishart distribution (n=500)
FFT Size              256
Steering vector       Formulation III, see :cite:`Sarradj2012`
===================== ========================================
