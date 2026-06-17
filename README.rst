================================================================================
AcouPipe
================================================================================

**AcouPipe** is a Python toolbox for generating *large-scale, configurable* microphone array datasets
for **acoustical source localization and characterization** with Acoular_. It is designed for
machine-learning workflows, enabling reproducible dataset generation with physically meaningful
signal models and measurement-based propagation.

Key features
============

- **On-the-fly dataset synthesis**
- **Multiple propagation backends**
  - free-field propagation
  - measured RIR-based propagation
- **Distributed computation** with Ray_ for scalable generation and feature extraction.
- **Efficient storage** of machine-learning features (rather than raw multichannel time data).


Documentation
=============

The full documentation (API reference, examples, and dataset descriptions) is available here:
`https://adku1173.github.io/acoupipe/ <https://adku1173.github.io/acoupipe/>`_.


Installation
============

AcouPipe is currently available as a source-only package and is not published on PyPI or Conda.
It supports Python ``>=3.11,<3.14``.

First, clone the repository:

.. code-block:: bash

   git clone https://github.com/adku1173/acoupipe.git
   cd acoupipe

We recommend installing AcouPipe inside a virtual environment.

Using ``uv``
------------

If you use `uv <https://docs.astral.sh/uv/#installation>`_, create an environment with:

.. code-block:: bash

   uv venv

Then install AcouPipe from source:

.. code-block:: bash

   uv pip install -e .

Using ``pip`` and ``venv``
--------------------------

Create and activate a virtual environment:

.. code-block:: bash

   python3 -m venv my-env
   source my-env/bin/activate  # Linux/macOS
   my-env\Scripts\activate     # Windows

Then install AcouPipe from source:

.. code-block:: bash

   pip install -e .


Datasets
========

AcouPipe provides three default dataset generators:

* **DatasetSynthetic**
  A simple and fast baseline that relies on synthetic (e.g., white-noise) source signals and spatially
  stationary sources under anechoic conditions. 

  .. figure:: docs/source/_static/msm_layout.png
     :width: 600
     :align: center

* **DatasetMIRACLE**
  Uses measured SRIRs from the `MIRACLE dataset <https://doi.org/10.14279/depositonce-20837>`_
  (TU Berlin anechoic chamber) combined with synthetic source signals, resulting in a realistic
  and quasi-infinite dataset.

  .. figure:: docs/source/_static/msm_miracle.png
     :width: 600
     :align: center

* **DatasetSRIRACHA**
  Uses measured SRIRs from the `SRIRACHA dataset <https://doi.org/10.14279/depositonce-23943>`_,
  recorded with the same planar microphone array as MIRACLE, but in a reverberant shoebox room with
  varying absorption conditions. The dataset exposes eight main measurement scenarios that combine two
  source-receiver plane distances, two source arrangements, and two room absorption settings.

  .. figure:: docs/source/_static/sriracha_t60-measurement.jpg
     :width: 600
     :align: center


Data generation philosophy
==========================

Instead of storing raw multichannel time signals, AcouPipe stores only the **machine-learning-relevant
features** required by the training pipeline (e.g., source maps, time data, cross-spectral matrices, etc.)
This reduces storage requirements and enables:

- generation of **portable** datasets of manageable size,
- **on-the-fly** data generation during training,
- **reproducibility** via stored random seeds and deterministic pipelines.

Quick start
===========

The dataset classes are Python iterables that yield one sample at a time (dictionary-based),
which integrates well with PyTorch / TensorFlow input pipelines. Simply choose one of the dataset generator classes.
Each class provides methods to store data directly to disk in HDF5 format or as TFRecords. In addition, AcouPipe
supports distributed dataset generation with Ray_, which allows to use multiple CPU cores or cluster nodes for
faster data generation.

.. code-block:: python

   from acoupipe.datasets.synthetic import DatasetSynthetic

   # instantiate dataset class 
   # The cross-spectral-matrix (CSM) used to calculate the sourcemap feature will be
   # sampled from a Wishart distribution (mode="wishart")
   # Computation for each dataset sample will be distributed over 2 tasks (tasks=2)
   dataset = DatasetSynthetic(mode="wishart", tasks=2)

   # generator that creates beamforming maps for 10 different source cases at a frequency of 2000 Hz 
   # additionally, extract source locations ("loc" feature)
   data_generator = dataset.generate(
      features=["sourcemap","loc"], size=10, f=[2000], num=0)
                                       
   data_sample = next(data_generator) # obtain first source case sample 
   print(data_sample.keys())  # dict_keys(['sourcemap', 'loc', 'f'])

For reverberant scenes with measured room acoustics, AcouPipe also provides
``DatasetSRIRACHA`` in ``acoupipe.datasets.experimental``. The scenario name encodes the room
configuration (``SR`` empty, ``SRA`` absorbent), the source-receiver plane distance (``1`` near,
``2`` far), and an optional dense local grid (``-D``).

.. code-block:: python

   from acoupipe.datasets.experimental import DatasetSRIRACHA

   srir_dir = None  # or path to local SRIRACHA HDF5 files

   dataset = DatasetSRIRACHA(
      scenario="SRA2-D",
      mode="wishart",
      srir_dir=srir_dir,
   )

   data_generator = dataset.generate(
      features=["sourcemap", "loc", "f"], split="training", size=10, f=[2000], num=0)

If ``srir_dir`` is left as ``None``, AcouPipe downloads the required SRIRACHA data into the cache
directory.


Performance benchmarks (DatasetSynthetic)
-----------------------------------------

The plots below show performance results for computationally demanding feature configurations of
``DatasetSynthetic``:

.. image:: docs/source/_static/compute4_all_features-over-tasks_DatasetSynthetic_f4000.png
   :width: 100%
   :align: center

.. image:: docs/source/_static/compute4_all_features-over-tasks_DatasetSynthetic_fNone.png
   :width: 100%
   :align: center


Citation
========

If you use AcouPipe and/or the associated datasets in scientific work, please cite:

- the AcouPipe framework paper,
- the MIRACLE dataset paper (if you use DatasetMIRACLE),
- the SRIRACHA dataset record (if you use DatasetSRIRACHA).

.. code-block:: bibtex

   @article{Kujawski2023,
     author  = {Kujawski, Adam and Pelling, Art J. R. and Jekosch, Simon and Sarradj, Ennes},
     title   = {A framework for generating large-scale microphone array data for machine learning},
     journal = {Multimedia Tools and Applications},
     year    = {2023},
     doi     = {10.1007/s11042-023-16947-w}
   }

   @article{Kujawski2024,
     author  = {Kujawski, Adam and Pelling, Art J. R. and Sarradj, Ennes},
     title   = {MIRACLE - a Microphone Array Impulse Response Dataset for Acoustic Learning},
     journal = {EURASIP Journal on Audio, Speech, and Music Processing},
     year    = {2024},
     volume  = {2024},
     number  = {1},
     pages   = {32},
     doi     = {10.1186/s13636-024-00352-8}
   }

   @misc{Pelling2025,
     title     = {{{SRIRACHA}}: {{Shoebox Room Impulse Response Archive}} with {{Varying Absorption}}},
     author    = {Pelling, Art J. R. and Kujawski, Adam and Sarradj, Ennes},
     year      = {2025},
     month     = jul,
     publisher = {Technische Universit\"at Berlin},
     doi       = {10.14279/DEPOSITONCE-23943}
   }


License
=======

- **AcouPipe (code)** is licensed under the BSD license. See the file ``LICENSE`` for details.
- **MIRACLE** and **SRIRACHA** are distributed under their respective dataset licenses. SRIRACHA is
  released under **CC BY-NC-SA 4.0**; please ensure compliance before using the measured SRIRs in
  downstream work.


.. Links:
.. _SLURM: https://slurm.schedmd.com/quickstart.html
.. _Singularity: https://sylabs.io/guides/3.0/user-guide/quick_start.html
.. _Ray: https://docs.ray.io/en/latest
.. _`Ray Cluster`: https://docs.ray.io/en/latest/cluster/getting-started.html
.. _Tensorflow: https://www.tensorflow.org/
.. _`Tensorflow Dataset API`: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
.. _TFRecord: https://www.tensorflow.org/tutorials/load_data/tfrecord
.. _DockerHub: https://hub.docker.com/r/adku1173/acoupipe/tags?page=1&ordering=last_updated
.. _Acoular: http://www.acoular.org
.. _HDF5: https://support.hdfgroup.org/documentation/index.html
.. _Pandas: https://pandas.pydata.org/docs/
.. _h5py: https://docs.h5py.org/en/stable/
.. _tqdm: https://github.com/tqdm/tqdm
