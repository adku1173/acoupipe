
Install
============

Dependencies
------------
The AcouPipe package depends on:

* Acoular_
* Ray_
* h5py_
* tqdm_
* Tensorflow_ (optional)


How to install
------------------

Download or clone the acoupipe repository and enter the directory, e.g. change "</path/to/dir>" to the desired path and execute:

.. code-block::

   DIR=</path/to/dir> && git clone git@github.com:adku1173/acoupipe.git $DIR  && cd $DIR && unset DIR


Next, install module with pip. This will install all necessary `dependencies`_:

.. code-block::

   pip install .


Optionally, also install Tensorflow with (requires pip >19.0):

.. code-block::
   
   pip install tensorflow
   

or follow the instructions in https://www.tensorflow.org/install.

