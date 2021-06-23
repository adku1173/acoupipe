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
    on_trait_change, Dict ,HasPrivateTraits, List

from acoular.h5files import H5FileBase ,_get_h5file_class
from os import path

from acoular import config
config.h5library = "h5py"

from .pipeline import DataGenerator

class BaseLoadDataset(DataGenerator):
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

    #: Names of features, is set automatically / read from file.
    features = List( 
        desc="names of the features in the dataset")
    
    #: Number of features, is set automatically / read from file.
    numfeatures = CLong(0, 
        desc="number of features in the dataset")
    
    #: Number of features, is set automatically / read from file.
    indices = List( 
        desc="the indices of the dataset")

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
            raise FileNotFoundError("No such file: %s" % self.name)
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        file = _get_h5file_class()
        self.h5f = file(self.name,mode="r")
        #self.load_dataset()
        self.load_metadata()

    def load_dataset( self ):
        """ loads dataset from .h5 file into the dataset attribute.
        Should only be used if dataset is small in memory. """
        for key in self.indices:
            self.dataset[key] = {}
            for feature in self.h5f[key].keys():
                self.dataset[key][feature] = self.h5f[key][feature]
                    
    def load_metadata( self ):
        """ loads metadata from .h5 file. Only for internal use. """
        self.metadata = {}
        indices = list(self.h5f.keys())
        if'metadata' in indices:
            indices.remove('metadata')
            for feature in self.h5f['/metadata'].keys():
                self.metadata[feature] = self.h5f['/metadata'][feature][()]
        int_indices = list(map(int,indices))
        int_indices.sort()
        self.indices = list(map(str,int_indices))
        self.numsamples=len(self.indices)        
        if self.numsamples > 0:
            self.numfeatures = len(self.h5f[self.indices[0]].keys()) # assumes the same number of features in every sample of the dataset
            self.features = list(self.h5f[self.indices[0]].keys())

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
                for i in range(1,self.numsamples+1):
                    yield {key:value[()] for key,value in self.h5f[str(i)].items()}
            else:
                for i in range(1,self.numsamples+1):
                    yield {
                        key:value[()] for key,value in self.h5f[str(i)].items() if key in features}
            return
        return sample_generator

    def get_data(self):
        """ 
        Python generator that iteratively yields the samples of the
        dataset in ascending sample index order (e.g. 1,2,...,N). 
        
        Returns
        -------
        Dictionary containing a sample of the data set 
        {feature_name[key] : feature[values]}. 
        """
        for i in range(1,self.numsamples+1):
            yield {key:value[()] for key,value in self.h5f[str(i)].items()}
