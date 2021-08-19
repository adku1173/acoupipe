.. _doc:


Documentation
=================

The AcouPipe module extends the computational 
pipeline-based concept of Acoular_ and provides additional 
tools that can be helpful to generate realizations 
of features in a predefined random process. 




Dependencies
------------
This package works with Python 3.7 or 3.8 and depends on:

* Acoular_
* Ray_
* Pandas_
* h5py_
* tqdm_
* Tensorflow_ (optional)


Installation
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

Module Overview
------------------

The following UML flowchart gives a rough overview of AcouPipe's 
classes and their inheritance relationships. 

.. figure:: ../_static/acoupipe_uml.png
    :width: 780


Sampler Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A manipulation of object characteristics according to a certain 
random distribution can be achieved by using the :code:`BaseSampler` derived classes included in the :code:`sampler.py` module. 
All :code:`BaseSampler` derived classes represent random processes that can be used to manipulate the attributes of Acoular's objects according to a specified distribution. 
A random process is defined by a random variable and a corresponding random state. Both properties are attributes of all :code:`BaseSampler` derived classes. 
AcouPipe offers a variety of different types of samplers in the :code:`sampler.py` module.
The random variable that can be passed to class instances of the sampler module must either be derived from or be part of the :code:`scipy.stats` module. 

This example illustrates how the RMS value of two white noise signals can be sampled according to a normal distribution. For this purpose, an instance of the :code:`BaseSampler` 
derived :code:`NumericAttributeSampler` class is used. The two :code:`WNoiseGenerator` objects are given as targets to the sampler object. 
New RMS values following a normal distribution are assigned to the :code:`WNoiseGenerator` objects each time the sample method of the :code:`NumericAttributeSampler` object is evaluated.    

.. code-block:: python

    import acoular
    import acoupipe
    from scipy.stats import norm

    random_var = norm(loc=1.,scale=.5)

    n1 = acoular.WNoiseGenerator( sample_freq=24000, 
                    numsamples=24000*5, 
                    rms=1.0,
                    seed=1 )

    n2 = acoular.WNoiseGenerator( sample_freq=24000, 
                    numsamples=24000*5, 
                    rms=.5,
                    seed=2 )

    rms_sampler = acoupipe.NumericAttributeSampler(
                    target=[n1,n2],
                    attribute='rms',
                    random_var=random_var,
                    random_state=10)

    rms_sampler.sample()


Pipeline Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

Classes defined in the :code:`pipeline.py` module have the ability to iteratively perform tasks on the related computational pipeline to build up a dataset. 
The results of these tasks are the features (and labels) associated with a specific sample of the dataset. 
Feature creation tasks can be specified by passing callable functions that are evoked at each iteration of the :code:`BasePipeline`'s :code:`get_data()` generator method. 
It is worth noting that such a data generator can also be used directly to feed a machine learning model without saving the data to file, as common machine learning frameworks, such as Tensorflow_, offer the possibility to consume data from Python generators.
Control of the state of the sampling process is maintained via the :code:`sampler` attribute holding a list of :code:`BaseSampler` derived instances. 

.. code-block:: python

    def calculate_csm(powerspectra):
        return powerspectra.csm

    pipeline = acoupipe.BasePipeline(
        sampler=[rms_sampler],
        numsamples = 5,
        features={'csm' : (calculate_csm, ps),}
        )
            
    data_generator = pipeline.get_data()


Writer Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :code:`writer.py` model provides classes to store the data extracted by the pipeline. 
The current implementation includes classes to save data in a container-like file format (.h5 file with the :code:`WriteH5Dataset` class) or binary format (.tfrecord file with the :code:`WriteTFRecord` class). 
The latter can be efficiently consumed by the Tensorflow framework for machine learning.

.. code-block:: python

    file_writer = acoupipe.WriteH5Dataset(
                source=pipeline,
                )
        
    file_writer.save()
    

Loader Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :code:`loader.py` module provides the :code:`LoadH5Dataset` class to load the datasets stored into .h5 files.

Examples
------------------

