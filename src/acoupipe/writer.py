"""Provides classes to store the data extracted by :class:`~acoupipe.pipeline.BasePipeline` derived classes.

Purpose of the Writer Module
----------------------------
The :code:`writer.py` module provides classes to store the data extracted by the pipeline. 
The current implementation includes classes to save data in a container-like file format (.h5 file with the :code:`WriteH5Dataset` class) or 
binary format (.tfrecord file with the :code:`WriteTFRecord` class). 
The latter can be efficiently consumed by the Tensorflow framework for machine learning.

.. code-block:: python

    file_writer = acoupipe.WriteH5Dataset(
                source=pipeline,
                )
        
    file_writer.save()
  

"""


from datetime import datetime
from os import path

from acoular import config
from h5py import File as H5File
from traits.api import Bool, Dict, File, Function, Instance, List, Str, Trait

from acoupipe.config import TF_FLAG
from acoupipe.pipeline import DataGenerator


class BaseWriteDataset(DataGenerator):
    """Base class intended to write data from :class:`~acoupipe.pipeline.BasePipeline` instances to a specific file format.
    
    This class has no functionality and should not be used.
    """
    
    #: source instance that has to be of type :class:`~acoupipe.pipeline.DataGenerator`
    source = Instance(DataGenerator)

    def save(self):
        """Saves data from a :class:`~acoupipe.pipeline.BasePipeline` instance specified at :attr:`source` to file."""
        # write to File...
        pass

    def get_data(self, progress_bar=True):
        """Python generator that saves source output data to file and passes the data to the next object.

        Parameters
        ----------
        progress_bar : bool
            If True, a progress bar is shown.

        Returns
        -------
        Dictionary containing a sample of the data set 
        {feature_name[key] : feature[values]}. 
        """
        for data in self.source.get_data(progress_bar):
            # write to File...
            yield data


class WriteH5Dataset(BaseWriteDataset):
    """Class intended to write data to a `.h5` file."""
       
    #: Name of the file to be saved. 
    name = File(filter=["*.h5"], 
                 desc="name of data file")   
           
    # #: Number of samples to write to file by :meth:`result` method. 
    # #: defaults to -1 (write as long as source yields data). 
    # numsamples_write = Int(-1)
    
    #: flag that can be raised to stop file writing when running in detached thread
    writeflag = Bool(True)
    
    #: a list with names of the features to be saved. By default [], 
    #: meaning that all features comming from the source will be saved. 
    features = List([
        ], desc="the names of the features to be saved")
    
    #: Metadata to be stored in HDF5 file object
    metadata = Dict(
        desc="metadata to be stored in .h5 file")

    def create_filename(self):
        if self.name == "":
            name = datetime.now().isoformat("_").replace(":","-").replace(".","_")
            self.name = path.join(config.td_dir,name+".h5")

    def get_file(self):
        self.create_filename()
        f5h = H5File(self.name, mode = "w")
        return f5h

    def get_filtered_features(self):
        if self.features:
            if "idx" not in self.features:
                subf = self.features.copy() + ["idx"]
            else:
                subf = self.features
            return subf

    def _add_data(self, f5h, data, subf):
        dataset_idx = str(data["idx"])
        #create a group for each Sample
        f5h.create_group(dataset_idx)
        #store dict in the group
        if not subf:
            [f5h.create_dataset(f"{dataset_idx}/{key}",data=value) for key, value in data.items()]
        else:
            [f5h.create_dataset(f"{dataset_idx}/{key}",data=value) for key, value in data.items() if key in subf]

    def _add_metadata(self, f5h):
        """adds metadata to .h5 file."""
        nitems = len(self.metadata.items())
        if nitems > 0:
            f5h.create_group("metadata")
            for key, value in self.metadata.items():
                f5h.create_dataset(f"metadata/{key}",data=value)

    def save(self, progress_bar=True):
        """Saves the output of the :meth:`get_data()` method of :class:`~acoupipe.pipeline.BasePipeline` to .h5 file format."""
        f5h = self.get_file()
        subf = self.get_filtered_features() 
        for data in self.source.get_data():
            self._add_data(f5h, data, subf)
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()

    def get_data(self, progress_bar=True):
        """Python generator that saves the data passed by the source to a `*.h5` file and yields the data to the next object.

        Returns
        -------
        Dictionary containing a sample of the data set 
        {feature_name[key] : feature[values]}. 
        """
        self.writeflag = True
        f5h = self.get_file()      
        subf = self.get_filtered_features() 
        for data in self.source.get_data(progress_bar): 
            if not self.writeflag: return     
            self._add_data(f5h, data, subf) 
            yield data
            f5h.flush()
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()
        


if TF_FLAG:
    import tensorflow as tf

    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def int_list_feature(value):
        """Returns an int64_list from a list od int values."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    def float_list_feature(value):
        """Returns a float_list from a list od float values."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


    class WriteTFRecord(BaseWriteDataset):
        """Class intended to write data from :class:`~acoupipe.pipeline.BasePipeline` to a .tfrecord.
        
        TFRecord files can be consumed by TensorFlow tf.data API. Stores data in binary format.
        """

        #: Name of the file to be saved. 
        name = File(filter=["*.tfrecords"], 
            desc="name of data file")   
        
        #: Dictionary with encoding functions (dict values) to convert data yielded by the pipeline to binary .tfrecord format.
        #: The key values of this dictionary are the feature names specified in the :attr:`features` attribute
        #: of the :attr:`source` object.         
        encoder_funcs = Dict(key_trait=Str(), value_trait=Function(), 
            desc="encoding functions to convert data yielded by the pipeline to binary format of .tfrecord file.")

        #: if True, writes an additional .txt file containing the names, types and shapes of the features stored in the
        #: tfrecord data set.   
        write_infofile = Bool(True, 
            desc="writes a file containing additional information about the stored features")

        #: Trait to set specific options to the .tfrecord file.
        options = Trait(None,tf.io.TFRecordOptions)

        def save(self, progress_bar=True):
            """Saves output of the :meth:`get_data()` method of :class:`~acoupipe.pipeline.BasePipeline` to .tfrecord format."""
            with tf.io.TFRecordWriter(self.name,options=self.options) as writer:
                for _i,features in enumerate(self.source.get_data(progress_bar)):
                    encoded_features = {n:self.encoder_funcs[n](f) for (n,f) in features.items() if self.encoder_funcs.get(n)}
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                    writer.flush()
                writer.close()
            
        def get_data(self, progress_bar=True):
            """Python generator that saves the data passed by the source to a `*.tfrecord` file and yields the data.

            Returns
            -------
            Dictionary containing a sample of the data set 
            {feature_name[key] : feature[values]}. 
            """
            with tf.io.TFRecordWriter(self.name,options=self.options) as writer:
                for _i,features in enumerate(self.source.get_data(progress_bar)):
                    encoded_features = {n:self.encoder_funcs[n](f) for (n,f) in features.items() if self.encoder_funcs.get(n)}
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                    yield features
                    writer.flush()
                writer.close()            
