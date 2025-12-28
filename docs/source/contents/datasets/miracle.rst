.. _dataset_miracle:

DatasetMIRACLE
==============

``DatasetMIRACLE`` is a microphone array dataset generator using experimentally measured spatial room impulse responses (SRIRs) from the `MIRACLE`_ dataset. The generator follows the same workflow as :class:`acoupipe.datasets.synthetic.DatasetSynthetic`, but uses measured transfer functions / impulse responses instead of analytic ones. Multi-source scenarios with possibly closing neighboring sources are realized by superimposing signals that have been convolved with the provided SRIRs.

.. figure:: ../../_static/msm_miracle.png
    :width: 750
    :align: center

    Measurement setup ``R2`` from the `MIRACLE`_ dataset.


Scenarios
---------

The MIRACLE dataset provides SRIRs from different measurement setups with the same microphone array, selectable via the :code:`scenario` parameter. The underlying measurement setup for :code:`scenario="R2"` is shown above.

.. list-table:: Available scenarios
    :header-rows: 1
    :widths: 5 10 10 10 10 10 10

    *   - Scenario
        - Download Size
        - Environment
        - c0
        - # SRIRs
        - Source-plane dist.
        - Spatial sampling
    *   - A1
        - 1.1 GB
        - Anechoic
        - 344.7 m/s
        - 4096
        - 73.4 cm
        - 23.3 mm
    *   - D1
        - 300 MB
        - Anechoic
        - 344.8 m/s
        - 4096
        - 73.4 cm
        - 5.0 mm
    *   - A2
        - 1.1 GB
        - Anechoic
        - 345.0 m/s
        - 4096
        - 146.7 cm
        - 23.3 mm
    *   - R2
        - 1.1 GB
        - Reflective Ground
        - 345.2 m/s
        - 4096
        - 146.7 cm
        - 23.3 mm


Default FFT parameters
----------------------

The underlying default FFT parameters are:

.. table:: FFT Parameters

    ===================== ========================================
    Sampling Rate         fs=32,000 Hz
    Block size            256 Samples
    Block overlap         50 %
    Windowing             von Hann / Hanning
    ===================== ========================================


Randomized properties
---------------------

Several properties of the dataset are randomized for each source case when generating the data. This includes the number of sources, their positions, and strength. Their respective distributions are closely related to :cite:`Herold2017`. Uncorrelated white noise is added to the microphone channels by default. Note that the source positions are sampled from a grid according to the spatial sampling of the MIRACLE dataset.

.. table:: Randomized properties

    ==================================================================   ===================================================
    No. of Sources                                                       Poisson distributed (:math:`\lambda=3`)
    Source Positions [m]                                                 Bivariate normal distributed (:math:`\sigma = 0.1688 d_a`)
    Source Strength (:math:`[{Pa}^2]` at reference position)               Rayleigh distributed (:math:`\sigma_{R}=5`)
    Relative Noise Variance                                              Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
    ==================================================================   ===================================================


Example
-------

.. code-block:: python

    from acoupipe.datasets.experimental import DatasetMIRACLE

    srir_dir = None  # optionally set a local path to MIRACLE SRIR files
    dataset = DatasetMIRACLE(scenario='A1', mode='wishart', srir_dir=srir_dir)

    dataset_generator = dataset.generate(size=10, f=2000, features=['sourcemap', 'loc', 'f'], split='training')

    data_sample = next(dataset_generator)

    import acoular as ac
    import matplotlib.pyplot as plt
    import numpy as np

    extent = dataset.config.grid.extend()

    # sound pressure level
    Lm = ac.L_p(data_sample['sourcemap']).T
    Lm_max = Lm.max()
    Lm_min = Lm.max() - 20

    # plot sourcemap
    plt.figure()
    plt.title(f'Beamforming Map (f={data_sample[\"f\"][0]} Hz, scenario={dataset.config.scenario})')
    plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin='lower')
    plt.colorbar(label='Sound Pressure Level (dB)')
    # plot source locations
    for loc in data_sample['loc'].T:
        plt.scatter(loc[0], loc[1])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

The generator yields one sample at a time as a dictionary, including helper fields ``idx`` and ``seeds`` to keep data generation reproducible when running in parallel.

API reference: :class:`acoupipe.datasets.experimental.DatasetMIRACLE`
