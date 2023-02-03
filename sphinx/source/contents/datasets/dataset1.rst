
Dataset1
========
This section will explain the default dataset1, which was used in [Kuj22]_.

The following figure illustrates the virtual measurement setup.


.. figure:: ../../_static/msm_layout.png
    :width: 780


The dataset is created by a simulation process with Acoular_ and comprises the following independent splits:

* training dataset  
* validation dataset 
* test dataset

The size of each split ('training', 'validation', 'test') can be freely chosen since the dataset is fully synthetic.
In [Kuj22]_ 50 million training samples and 10,000 validation samples were used.
The number of sources was uniformly distributed in the training data.

Instead of raw time-data, only the necessary input features for acoustical source characterization are stored.
This allows the user to create datasets of manageable size that are portable and facilitate reproducible research.
Depending on the users choice, the dataset comprises the following input features:

* Cross-Spectral Matrix / non-redundant Cross-Spectral Matrix (e.g. in [Cas21]_)
* Conventional Beamforming Map (e.g. in [Kuj19]_)
* Eigenmodes (e.g. in [Kuj22]_)


Dataset Characteristics
-------------------------

**fixed characteristics:**

===================== ========================================  
Environment           Unechoic, Resting, Homogeneous Fluid
Microphone Array      Vogel's spiral, M=64, Aperture Size 1m
Observation Area      x,y in [-0.5,0.5], z=0.5
Source Type           Monopole 
Source Signals        Uncorrelated White Noise (T=5s)
Sampling Rate         He = 40, f=13720 Hz 
No. of Time Samples   68.600 
===================== ========================================

**sampled characteristics:**

==================================================================   ===================================================  
Sensor Position Deviation [m]                                        Normal Distributed (sigma = 0.001)
No. of Sources                                                       Poisson Distributed (lambda=3)
Source Positions                                                     Normal Distributed (sigma = 0.1688) 
Source Strength (:math:`{Pa}^2` at reference microphone)             Rayleigh Distributed (sigma_R=5)
==================================================================   ===================================================

Input Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can save one of the three different input features to file:

* **Cross-Spectral Matrix (CSM):** :code:`'csm'` of shape: (B,64,64,2)
* **non-redundant Cross-Spectral Matrix:** :code:`'csmtriu'` of shape: (B,64,64,1)
* **Conventional Beamforming Map:** :code:`'sourcemap'` of shape: (B,64,64,1)

The first axis of each feature corresponds to the B FFT coefficients. The non-redundant CSM follows the 
approach stated in [Cas21]_ (the conjugate complex of the normal CSM is neglected). 
The underlying processing parameters used to calculate the CSM and/or the source map are:

===================== ========================================  
Block size            128 Samples
Block overlap         50 %
Windowing             von Hann / Hanning
Steering vector       Formulation 3, see [Sar12]_
===================== ========================================

Labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset comprises labels for each source case:

**Source strength at the reference microphone:** :code:`'p2'`

The averaged squared sound pressure value at the reference microphone position (red dot) is
stored as an estimate of the source strength for each individual source and frequency.
A value of zero is stored for non-existing sources. With a maximum number of 10 possible sources, this results 
in an array of shape (65,J) per case, whereby J refers to the number of sources present. 
It should be noted that the entries are sorted in descending order according to the overall RMS value of the source signal. 
The descending order is not strictly maintained when only a single frequency coefficient is considered.

**Source location:** :code:`'loc'`

The location in the x,y plane of each source is stored. The source location array is of shape (3,J). 
The source order is the same as for the source strength estimate :code:`p2`.

**Number of sources:** :code:`'nsources'`

An integer providing the number of sources.

**Sample index:** :code:`'idx'`

The index referencing the sampled case in the dataset (starts at 1). 

**Involved random seeds:** :code:`'seeds'`

A list with random seeds for each object that performs a random sampling of dataset properties.
The combination is unique for each source case in the dataset. This makes it possible to re-simulate any 
specific sample of the dataset.

File Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user can save the data to two different file formats (HDF5_ or TFRecord_). 
It is recommended to use the .h5 file format.

**HDF5 format**

HDF5_ is a container-like format storing data in hierarchical order. 
Each case and the corresponding data is stored into a separate group of the file. 
The sample index acts as the group header. 
An additional :code:`metadata` group includes important metadata (e.g. sampling frequency, FFT block size, ...).

.. code-block:: bash

    └──'1'
        |── 'csm' (or 'sourcemap', or 'csmtriu') 
        |── 'loc' 
        |── 'p2'  
        |── 'nsources'
        |── 'seeds'
    └──'2'
        |── 'csm' 
        |── 'loc' 
        |── 'p2'  
        |── 'nsources'
        |── 'seeds'
    └──...
        |   ...
        |  
    └──'metadata'
        |   'sample_freq'
        |   ...

The correct order is always maintained, which is important when multiple source cases are simulated in parallel tasks.

**TFRecord format**

The TFRecord_ file format is a binary file format to store sequences of data developed by Tensorflow_. 
In case of running the simulation with multiple CPU threads, the initial sampling order of the source cases may not be maintained in the file. 
The exact case number can be reconstructed with the :code:`idx` and :code:`seeds` features when the file is parsed.