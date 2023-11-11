================================================================================
AcouPipe
================================================================================

**AcouPipe** :cite:`Kujawski2023` is a Python toolbox for generating unique acoustical source localization and characterization datasets with Acoular_ :cite:`Sarradj2017` that can be used for training of deep neural networks and machine learning. 

AcouPipe supports distributed computation with Ray_ and comes with a default configuration dataset inside a pre-built Docker container that can be downloaded from DockerHub_.


.. toctree::
   :maxdepth: 1
   :caption: Contents

   contents/install.rst
   contents/datasets.rst
   contents/examples.rst
   contents/apidoc.rst
   contents/lit.rst

Data Generation 
===============

Currently, AcouPipe provides two default classes to generate microphone array datasets:

* **DatasetSynthetic1** relies on synthetic source signals from which the features are extracted and has been used in the following publications: :cite:`Kujawski2019`, :cite:`Kujawski2022`, :cite:`Feng2022`.

* **DatasetMIRACLE** relies on synthetic source signals and measured spatial room impulse responses, acquired at the TU Berlin anechoic chamber :cite:`Kujawski2023a`.

See the latest performance benchmarks on `DatasetSynthetic1` for the most computational demanding features:

.. image:: _static/compute4_all_features-over-tasks_DatasetSynthetic1_f4000.png
   :width: 100%
   :align: center

.. image:: _static/compute4_all_features-over-tasks_DatasetSynthetic1_fNone.png
   :width: 100%
   :align: center

Citation 
========

Users can cite the package in their contributions by referring to :cite:`Kujawski2023`.
Here is an example citation in BibTeX format:

.. code-block:: bibtex

   @article{Kujawski2023,
   author = {Kujawski,Adam and Pelling, Art J. R. and Jekosch, Simon and Sarradj,Ennes},
   title = {A framework for generating large-scale microphone array data for machine learning},
   journal = {Multimedia Tools and Applications},
   year = {2023},
   doi = {10.1007/s11042-023-16947-w}
   }



