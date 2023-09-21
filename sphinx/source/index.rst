================================================================================
AcouPipe
================================================================================

**AcouPipe** :cite:`Kujawski2023` is a Python toolbox for generating unique acoustical source localization and characterization datasets with Acoular_ :cite:`Sarradj2017` that can be used for training of deep neural networks and machine learning. 
Existing datasets primarily addresses the development of microphone array processing algorithms for acoustic testing. 

AcouPipe supports distributed computation with Ray_ and comes with a default configuration dataset inside a pre-built Docker container that can be downloaded from DockerHub_.


.. toctree::
   :maxdepth: 1
   :caption: Contents

   contents/install.rst
   contents/datasets.rst
   contents/apidoc.rst
   contents/examples.rst
   contents/lit.rst
   
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




