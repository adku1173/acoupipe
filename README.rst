|python-version|

================================================================================
AcouPipe
================================================================================

**AcouPipe** is an easy-to-use Python toolbox for generating unique acoustical source localization and characterization data sets with Acoular_ that can be used for training of deep neural networks and machine learning. Instead of raw time-data, only the necessary input features for acoustical beamforming are stored, which include:

* Cross-Spectral Matrix / non-redundant Cross-Spectral Matrix (e.g. in [Cas21]_)
* Conventional Beamforming Map (e.g. in [Kuj19]_)

This allows the user to create data sets of manageable size that are portable and facilitate reproducible research.

AcouPipe supports distributed computation with Ray_ and comes with a default configuration data set inside a pre-built Docker container that can be downloaded from DockerHub_.

Documentation can be found at `https://adku1173.github.io/acoupipe/`_.