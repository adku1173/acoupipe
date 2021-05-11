# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2020-2023, Adam Kujawski, Simon Jekosch, Art Pelling, Ennes Sarradj.
#------------------------------------------------------------------------------

"""
Provides classes to load the datasets stored with :class:`~acoupipe.writer.BaseWriteDataset` derived classes. 
Currently, only the loading of data stored in .h5 files is possible.

.. autosummary::
    :toctree: generated/

    BaseLoadDataset
    LoadH5Dataset

"""

from traits.api import   Property,\
cached_property,  CLong, File, Instance, \
    on_trait_change, Dict ,HasPrivateTraits

from acoular.h5files import H5FileBase ,_get_h5file_class
from os import path

from acoular import config
config.h5library = "h5py"



class BaseLoadDataset(HasPrivateTraits):
    """
    Base class for all derived classes intended to load 
    data stored with instances of type :class:`~acoupipe.writer.BaseWriteDataset`.
    This class has no functionality and should not be used.
    """

    #:Full name of the .h5 file with data.
    name = File(filter=['*'], 
        desc="name of data file")

    def load_data(self):
        # load a File...
        pass



class LoadH5Dataset(BaseLoadDataset):
    
    """
    Loads data sets stored into `*.h5` file format.
    
    This class loads data from .h5 files and
    and provides information like the number of 
    samples (:attr:`numsamples`).
    """

    #: Full name of the .h5 file with data.
    name = File(filter=['*.h5'], 
        desc="name of data file")

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property( depends_on = 'name', 
        desc="basename of data file")
    
    #: The  data as array.
    dataset = Dict(
        desc="the actual time data array")
    
    #: Number of data samples, is set automatically / read from file.
    numsamples = CLong(0, 
        desc="number of samples in the dataset")
    
    #: Number of features, is set automatically / read from file.
    numfeatures = CLong(0, 
        desc="number of features in the dataset")
    
    #: HDF5 file object
    h5f = Instance(H5FileBase, transient = True)
    
    #: Provides metadata stored in HDF5 file object
    metadata = Dict(
        desc="metadata contained in .h5 file")

    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]
    
    @on_trait_change('basename')
    def load_data( self ):
        """ 
        Open the .h5 file and set attributes.
        """
        if not path.isfile(self.name):
            # no file there
            self.numsamples = 0
            self.numfeatures = 0
            raise IOError("No such file: %s" % self.name)
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        file = _get_h5file_class()
        self.h5f = file(self.name)
        self.load_dataset()
        self.load_metadata()

    def load_dataset( self ):
        """ loads dataset from .h5 file. Only for internal use. """
        
        for key in self.h5f.keys():
            if key != 'metadata':
                self.dataset[key] = {}
                self.numfeatures = len(self.h5f[key].keys())
                for feature in self.h5f[key].keys():
                    self.dataset[key][feature] = self.h5f[key][feature][()]
        
        if 'metadata' in self.h5f.keys():
            self.numsamples = len(self.h5f.keys())-1
        else:
            self.numsamples = len(self.h5f.keys())
            
        
    def load_metadata( self ):
        """ loads metadata from .h5 file. Only for internal use. """
        self.metadata = {}
        for key in self.h5f.keys():
            if key == 'metadata':
                for feature in self.h5f['/metadata'].keys():
                    self.metadata[feature] = self.h5f['/metadata'][feature][()]
        

    def get_dataset_generator(self, features=[]):
        """Creates a callable that returns a generator object. 
        
        This object can be used in conjunction with the Tensorflow `tf.data.Dataset` API to create
        a data generator with the `from_generator`method to feed machine learning models.

        Example to create a repeatable data set with the Tensorflow  `tf.data.Dataset` API:

        >>> h5data = LoadH5Dataset(name="some_dataset.h5")
        >>> generator = h5data.get_dataset_generator(features=['p2'])
        >>> output_signature = {
        ...    'p2' : tf.TensorSpec(shape=(16,), dtype=tf.float32),
        ... }
        >>>
        >>> dataset = tf.data.Dataset.from_generator(generator,
        ...     output_signature=output_signature).repeat()
        >>> p2 = next(iter(dataset)) # return p2

        Parameters
        ----------
        features : list, optional
            a list with names of the features to be yielded by the generator, by default [], 
            meaning that all features will be considered. 

        Returns
        -------
        callable
            A callable that returns a generator object 
        """
        def sample_generator():
            if not features:
                for data in self.dataset:
                    yield self.dataset[data]
            else:
                for data in self.dataset:
                    yield {
                        key:value for (key,value) in self.dataset[data].items() if key in features}
            return
        return sample_generator
