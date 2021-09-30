|python-version| |DOI|

================================================================================
AcouPipe
================================================================================

**AcouPipe** is an easy-to-use Python toolbox for generating unique acoustical source localization and characterization data sets with Acoular_ that can be used for training of deep neural networks and machine learning. Instead of raw time-data, only the necessary input features for acoustical beamforming are stored, which include:

* Cross-Spectral Matrix / non-redundant Cross-Spectral Matrix (e.g. in [Cas21]_)
* Conventional Beamforming Map (e.g. in [Kuj19]_)

This allows the user to create data sets of manageable size that are portable and facilitate reproducible research.

AcouPipe supports distributed computation with Ray_ and comes with a default configuration data set inside a pre-built Docker container that can be downloaded from DockerHub_.

Documentation can be found `here <https://adku1173.github.io/acoupipe/>`_.

.. figure:: _static/msm_layout.png
    :width: 780

    Virtual Measurement Setup of Example Dataset


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
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5176234.svg
   :target: https://doi.org/10.5281/zenodo.5176234
    
Literature
==========================

.. [Sar12] Sarradj, Ennes: Three-dimensional acoustic source mapping with different beamforming steering vector formulations. Advances in Acoustics and Vibration, pages 1–12, 2012.
.. [Cas21] Paolo Castellini, Nicola Giulietti, Nicola Falcionelli, Aldo Franco Dragoni, Paolo Chiariotti, A neural network based microphone array approach to grid-less noise source localization, Applied Acoustics, Volume 177, 2021, 107947, ISSN 0003-682X, https://doi.org/10.1016/j.apacoust.2021.107947.
.. [Kuj19] Adam Kujawski, Gert Herold, and Ennes Sarradj , "A deep learning method for grid-free localization and quantification of sound sources", The Journal of the Acoustical Society of America 146, EL225-EL231 (2019) https://doi.org/10.1121/1.5126020
