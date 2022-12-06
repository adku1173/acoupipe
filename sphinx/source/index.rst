|python-version|

================================================================================
AcouPipe
================================================================================

**AcouPipe** is an easy-to-use Python toolbox for generating unique acoustical source localization and characterization datasets with Acoular_ that can be used for training of deep neural networks and machine learning. Instead of raw time-data, only the necessary input features for acoustical beamforming are stored, which include:

* Cross-Spectral Matrix / non-redundant Cross-Spectral Matrix (e.g. in [Cas21]_)
* Conventional Beamforming Map (e.g. in [Kuj19]_)

This allows the user to create datasets of manageable size that are portable and facilitate reproducible research.

Acoupipe has beed used in the following publications: [Kuj19]_, [Kuj22]_, [Fen22]_.

AcouPipe supports distributed computation with Ray_ and comes with a default configuration dataset inside a pre-built Docker container that can be downloaded from DockerHub_.

Contents
========

.. toctree::
   :maxdepth: 1

   contents/dataset.rst
   contents/documentation.rst
   contents/example.rst
   contents/literature.rst
   api_ref/index_apiref.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

