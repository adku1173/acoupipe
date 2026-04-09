.. _store:

Store Datasets 
==============

The user can also save the datasets in two different file formats (HDF5_ or TFRecord_). 

HDF5 format
-----------

HDF5_ is a container-like format storing data in hierarchical order. 
Each case and the corresponding data is stored into a separate group of the file. 
The sample index acts as the group header. 

.. code-block:: bash

    └──'0'
        |── 'sourcemap'
        |── 'loc' 
        |── 'f' 
        |── 'seeds'
    └──'1'
        |── 'sourcemap'
        |── 'loc' 
        |── 'f' 
        |── 'seeds'
    └──...
        |   ...

The correct order is always maintained, which is important when multiple source cases are simulated in parallel tasks.

The following code snippet shows how to store the data in HDF5 format:

.. ipython::
    :okwarning:

    In [1]: from acoupipe.datasets.synthetic import DatasetSynthetic
    
    In [2]: dataset = DatasetSynthetic()
       ...: dataset.save_h5(features=['sourcemap','loc', 'f'], 
       ...:                 split='training', size=10, f=[2000], num=0, 
       ...:                 name='/tmp/training_dataset.h5')

A more in-depth example on how to save and load the data, and how to create a TensorFlow-ready pipeline from file can be found in the :ref:`Save and load datasets stored in HDF5 file format <Examples>` example.

TFRecord format
---------------

The TFRecord_ file format is a binary file format to store sequences of data developed by Tensorflow_. 
In case of running the simulation with multiple CPU threads, the initial sampling order of the source cases may not be maintained in the file. 
However, the exact source case number can be figured out with the :code:`idx` and :code:`seeds` features when the file is parsed.

The following code snippet shows how to store the data in TFRecord format:

.. ipython::
    :okwarning:

    In [3]: dataset.save_tfrecord(features=['sourcemap','loc', 'f'], 
       ...:                       split='training', size=10, f=[2000], num=0, 
       ...:                       name='/tmp/training_dataset.tfrecord')

A more in-depth example on how to save and load the data, and how to create a TensorFlow-ready pipeline from file can be found in the :ref:`Save and load datasets stored in TFRecord file format <Examples>` example.
