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
* :ref:`example`
* :ref:`lit`
* :ref:`manual`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`