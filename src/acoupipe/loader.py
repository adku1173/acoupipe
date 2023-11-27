"""Provides classes to load the datasets stored with :class:`~acoupipe.writer.BaseWriteDataset` derived classes."""
from os import path

from acoular import config
from h5py import File as H5File
from traits.api import CLong, Dict, File, Instance, List, Property, cached_property, on_trait_change

from acoupipe.pipeline import DataGenerator

config.h5library = "h5py"



class BaseLoadDataset(DataGenerator):
    """Base class for all derived classes intended to load data stored by :class:`~acoupipe.writer.BaseWriteDataset`.

    This class has no functionality and should not be used.
    """

    #:Full name of the file with data.
    name = File(filter=["*"],
        desc="name of data file")

    def load_data(self):
        """Open a dataset file and set attributes."""
        pass



class LoadH5Dataset(BaseLoadDataset):
    """Loads data sets stored into `*.h5` file format.

    This class loads data from `*.h5` files and
    provides information like the number of
    samples (:attr:`numsamples`).
    """

    #: Full name of the .h5 file with data.
    name = File(filter=["*.h5"],
        desc="name of data file")

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property( depends_on = "name",
        desc="basename of data file")

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
    h5f = Instance(H5File, transient = True)

    #: Provides metadata stored in HDF5 file object
    metadata = Dict(
        desc="metadata contained in .h5 file")

    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]

    @on_trait_change("basename")
    def load_data( self ):
        """Open the .h5 file and set attributes."""
        if not path.isfile(self.name):
            # no file there
            self.numsamples = 0
            self.numfeatures = 0
            raise FileNotFoundError("No such file: %s" % self.name)
        if self.h5f is not None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = H5File(self.name,mode="r")
        self.load_metadata()

    def load_metadata( self ):
        """Load metadata from .h5 file. Only for internal use."""
        self.metadata = {}
        indices = list(self.h5f.keys())
        if"metadata" in indices:
            indices.remove("metadata")
            for feature in self.h5f["/metadata"].keys():
                self.metadata[feature] = self.h5f["/metadata"][feature][()]
        int_indices = list(map(int,indices))
        int_indices.sort()
        self.indices = list(map(str,int_indices))
        self.numsamples=len(self.indices)
        if self.numsamples > 0:
            self.numfeatures = len(self.h5f[self.indices[0]].keys())
            self.features = list(self.h5f[self.indices[0]].keys())

    def get_dataset_generator(self, features=None):
        """Create a callable that returns a generator object.

        This object can be used in conjunction with the Tensorflow `tf.data.Dataset` API to create
        a data generator with the :meth:`from_generator` method of the `Tensorflow Dataset API`_
        to feed machine learning models.

        Example to create a repeatable data set with the Tensorflow `tf.data.Dataset` API is given in


        >>> h5data = LoadH5Dataset(name="some_dataset.h5")
        >>> generator = h5data.get_dataset_generator(features=['loc'])
        >>> output_signature = {
        ...    'loc' : tf.TensorSpec(shape=(3,None), dtype=tf.float32),
        ... }
        >>>
        >>> dataset = tf.data.Dataset.from_generator(generator,
        ...     output_signature=output_signature).repeat()
        >>> loc = next(iter(dataset)) # return locations

        Parameters
        ----------
        features : list, optional
            a list with names of the features to be yielded by the generator, by default None,
            meaning that all features will be considered.

        Returns
        -------
        callable
            A callable that returns a generator object
        """
        def sample_generator():
            indices = list(self.h5f.keys())
            if features is None:
                for idx in indices:
                    if idx != "metadata":
                        data = {key:value[()] for key,value in self.h5f[idx].items()}
                        data.update({"idx":idx})
                        yield data
            else:
                for idx in indices:
                    if idx != "metadata":
                        data = {key:value[()] for key,value in self.h5f[idx].items() if key in features}
                        data.update({"idx":idx})
                        yield data
            return
        return sample_generator

    def get_data(self):
        """Python generator that iteratively yields the samples of the dataset in ascending sample index order (e.g. 1,2,...,N).

        Returns
        -------
        dict
            Dictionary containing a sample of the dataset
            {feature_name[key] : feature[values]}
        """
        indices = list(self.h5f.keys())
        for idx in indices:
            if idx != "metadata":
                data = {key:value[()] for key,value in self.h5f[idx].items()}
                data.update({"idx":idx})
                yield data
