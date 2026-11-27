
.. _quickstart:

Quickstart
==========

Generate Datasets 
-----------------

To generate microphone array data with one of the available dataset generation methods, the user needs to import and instantiate the respective class and call the method :meth:`generate` with the desired parameters. The :meth:`generate` method returns a Python generator, which can be used to extract the desired features.

The following code snippet shows how to create a dataset generator:

.. ipython::
    :okwarning:

    In [1]: from acoupipe.datasets.synthetic import DatasetSynthetic
    
    In [2]: # instantiate dataset class
       ...: dataset = DatasetSynthetic()

    In [3]: # create data generator for frequency 2000 Hz (single frequency)
       ...: data_generator = dataset.generate(
       ...:     size=10, split='training', features=['sourcemap','loc', 'f'], f=[2000], num=0)

The user has to specify the :code:`size` of the dataset, the dataset split :code:`split` and the :code:`features` to be extracted. A full list of available features can be found in the :ref:`Features section <features>`. Optionally, the user can specify certain frequencies :code:`f` and the desired bandwidth :code:`num`. If no frequencies are given, the features will be extracted for all frequencies in the range :math:`[0, fs/2]` Hz.

.. admonition:: Dataset Splits

    All datasets comprise the following independent splits:

    * :code:`training` dataset
    * :code:`validation` dataset 
    * :code:`test` dataset

    The size of each split can be freely chosen.
    

In this example, we choose the :code:`sourcemap` feature, which is a 2D beamforming map with the squared sound pressure amplitudes at a reference position as values. Furthermore, we include the source locations :code:`loc` and frequencies :code:`f` as potential labels of the dataset. 

There are a few things to note when extracting features with the data generator: 

.. ipython::
    :okwarning:

    In [4]: data_sample = next(data_generator)

First, the data generator returns a single sample at a time, consisting of a dictionary with the feature names as keys and the corresponding data as values. Second, the :code:`data_sample` dictionary includes the additional item :code:`idx`, the sample index. Together with the dataset seed and the split, it determines the sample (see :ref:`reproducibility`) and identifies it in multi-processing scenarios where the sample order may not be maintained.

.. note::
    The frequency included in the data might be slightly different from the specified frequency. This is usually indicated by a warning and is due to the fact that the frequency is chosen from a discrete set of frequencies, which depends on the parameters of the FFT and the sampling rate :code:`fs` of the dataset. 
    
The sampling rate and the FFT parameters can be adjusted the following way:

.. ipython::
    :okwarning:

    In [5]: fs = 44100
       ...: block_size = 4096
       ...: dataset = DatasetSynthetic(fs=fs)
       ...: dataset.config.fft_params['block_size'] = block_size

Now, the data can be plotted, e.g. by using the `matplotlib`_ library:

.. ipython::
    :okwarning:
    :okexcept:

    In [6]: import acoular as ac
       ...: import matplotlib.pyplot as plt
       ...: import numpy as np

    In [7]: extent = dataset.config.grid.extent

    In [8]: # sound pressure level
       ...: Lm = ac.L_p(data_sample['sourcemap']).T
       ...: Lm_max = Lm.max()
       ...: Lm_min = Lm.max() - 20

    @savefig quickstart_sourcemap.png width=600px
    In [9]: # plot sourcemap
       ...: plt.figure()
       ...: plt.title(f'Beamforming Map (f={data_sample["f"][0]} Hz)')
       ...: plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin='lower')
       ...: plt.colorbar(label='Sound Pressure Level (dB)')
       ...: # plot source locations
       ...: for loc in data_sample['loc'].T:
       ...:     plt.scatter(loc[0], loc[1])
       ...: plt.xlabel('x (m)')
       ...: plt.ylabel('y (m)')


.. _reproducibility:

Reproducibility
---------------

Each sample is determined by the dataset :code:`seed` (default: 0), the :code:`split` name and the sample
index :code:`idx`. Generating a split again yields identical samples, independent of the number of parallel
tasks and of :code:`start_idx`. A single sample can be regenerated with
:code:`generate(..., start_idx=idx, size=1)`.

.. code-block:: python

    dataset = DatasetSynthetic(seed=1)  # an independent realization of all splits

Different split names (e.g. :code:`'training'`, :code:`'validation'`, :code:`'test'` or custom names) and
different seeds yield statistically independent samples. Each sampler draws from its own random stream, so
switching a sampler off (e.g. :code:`mic_pos_noise=False`) does not change the values drawn by the others.

.. _parallel-generation:

Parallel Generation
-------------------

With :code:`tasks > 1`, the samples are calculated in parallel by :code:`tasks` Ray worker processes.
Each worker calculates up to :code:`samples_per_task` samples (default: 16) per call, which reduces the
scheduling overhead for samples that are fast to compute. The sample order is not preserved; the
:code:`idx` item identifies each sample.

.. code-block:: python

    dataset = DatasetSynthetic(tasks=8, samples_per_task=16)

.. warning::

    Each worker process should use a single thread. Set :code:`NUMBA_NUM_THREADS=1`,
    :code:`OMP_NUM_THREADS=1`, :code:`OPENBLAS_NUM_THREADS=1` and :code:`MKL_NUM_THREADS=1` in the
    environment before starting Python or the Ray cluster (:code:`ray start`). Otherwise, every worker
    starts as many threads as there are CPU cores, and the workers slow each other down severely.

.. warning::

    Acoular compiles its Numba functions when it is imported and caches them on disk. If several
    workers compile into an empty cache at the same time, the cache can become inconsistent and the
    workers crash with :code:`LLVM ERROR: Symbol not found`. On a single machine, the main process
    fills the cache before the workers start, as long as the workers use the same Python environment
    (see :ref:`known-issues` for :code:`uv run`). On clusters, fill a node-local cache on every node
    before starting the Ray workers, see :ref:`hpc-singularity`.

A more in-depth example on how to generate the data, and how to create a TensorFlow-ready pipeline that produces data on the fly can be found in the :ref:`Model training with training data generated on the fly <Examples>` example.
