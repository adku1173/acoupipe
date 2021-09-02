|python-version|

================================================================================
AcouPipe
================================================================================

**AcouPipe** is an easy-to-use Python toolbox for generating unique acoustical source localization and characterization datasets with Acoular_ that can be used for training of deep neural networks and machine learning. Instead of raw time-data, only the necessary input features for acoustical beamforming are stored, which include:

* Cross-Spectral Matrix / non-redundant Cross-Spectral Matrix (e.g. in [Cas21]_)
* Conventional Beamforming Map (e.g. in [Kuj19]_)

This allows the user to create datasets of manageable size that are portable and facilitate reproducible research.

AcouPipe supports distributed computation with Ray_ and comes with a default configuration dataset inside a pre-built Docker container that can be downloaded from DockerHub_.

Contents
========

* :ref:`data`
* :ref:`doc`
* :ref:`lit`
* :ref:`manual`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Links:
.. _SLURM: https://slurm.schedmd.com/quickstart.html
.. _Singularity: https://sylabs.io/guides/3.0/user-guide/quick_start.html
.. _Ray: https://docs.ray.io/en/master/
.. _`Ray Cluster`: https://docs.ray.io/en/master/cluster/index.html
.. _Tensorflow: https://www.tensorflow.org/
.. _`Tensorflow Dataset API`: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
.. _TFRecord: https://www.tensorflow.org/tutorials/load_data/tfrecord
.. _DockerHub: https://hub.docker.com/r/adku1173/acoupipe/tags?page=1&ordering=last_updated
.. _Acoular: http://www.acoular.org
.. _HDF5: https://portal.hdfgroup.org/display/HDF5/HDF5
.. _Pandas: https://pandas.pydata.org/docs/
.. _h5py: https://docs.h5py.org/en/stable/
.. _tqdm: https://github.com/tqdm/tqdm

.. Badges:
.. |python-version| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8-blue
   :target: https://www.python.org/