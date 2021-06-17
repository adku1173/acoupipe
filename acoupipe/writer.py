# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2020-2023, Adam Kujawski, Simon Jekosch, Art Pelling, Ennes Sarradj.
#------------------------------------------------------------------------------

"""
Provides classes to store the data extracted by :class:`~acoupipe.pipeline.BasePipeline` derived classes. 
Current implementation includes :class:`WriteH5Dataset` to save data to .h5 files and :class:`WriteTFRecord` to save .tfrecord files.
The latter can be efficiently consumed by the `Tensorflow <https://www.tensorflow.org//>`_ framework for machine learning and is optional. 

.. autosummary::
    :toctree: generated/

    BaseWriteDataset
    WriteH5Dataset
    WriteTFRecord

"""


from traits.api import Instance, Dict, Str, File,\
    Function, Trait, Bool, List
from .pipeline import DataGenerator

from acoular.h5files import  _get_h5file_class
from acoular import config

from datetime import datetime
from os import path
from numpy import ndarray

TF_FLAG = True
try:
    import tensorflow as tf
except:
    TF_FLAG = False
    print('No tensorflow package installed!')
    
    
class BaseWriteDataset(DataGenerator):
    """Base class for all derived classes intended to write 
    data from :class:`~acoupipe.pipeline.BasePipeline` instances
    to a specific file format. This class has no functionality 
    and should not be used.
    """
    
    #: source instance that has to be of type :class:`~acoupipe.pipeline.DataGenerator`
    source = Instance(DataGenerator)

    def save(self):
        """saves data from a :class:`~acoupipe.pipeline.BasePipeline` 
        derived class instance specified at :attr:`source` to file.
        """
        # write to File...
        pass

    def get_data(self):
        """ 
        Python generator that saves source output data to file and
        passes the data to the next object.
      
        Returns
        -------
        Dictionary containing a sample of the data set 
        {feature_name[key] : feature[values]}. 
        """
        for data in self.source.get_data():
            # write to File...
            yield data


class WriteH5Dataset(BaseWriteDataset):
    """
    Class intended to write data to a `.h5` file.
    """
       
    #: Name of the file to be saved. 
    name = File(filter=['*.h5'], 
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
        if self.name == '':
            name = datetime.now().isoformat('_').replace(':','-').replace('.','_')
            self.name = path.join(config.td_dir,name+'.h5')

    def get_initialized_file(self):
        file = _get_h5file_class()
        self.create_filename()
        f5h = file(self.name, mode = 'w')
        return f5h

    def get_filtered_features(self):
        if self.features:
            if not 'idx' in self.features:
                subf = self.features.copy() + ['idx']
            else:
                subf = self.features
            return subf
        
    def save(self):
        """ 
        saves the output of the :meth:`get_data()` method of a :class:`~acoupipe.pipeline.BasePipeline` 
        derived class instance specified at :attr:`source` to .h5 file format.  
        """
        f5h = self.get_initialized_file()
        subf = self.get_filtered_features() 
        for data in self.source.get_data():
            #create a group for each Sample
            ac = f5h.create_new_group(str(data['idx']))
            #store dict in the group
            if not subf:
                [f5h.create_array(ac,key, value) for key, value in data.items()]   
            else:
                [f5h.create_array(ac,key, value) for key, value in data.items() if key in subf]   
        self.add_metadata(f5h)
        f5h.flush()
        f5h.close()

    def add_metadata(self, f5h):
        """ adds metadata to .h5 file """
        nitems = len(self.metadata.items())
        if nitems > 0:
            ac = f5h.create_new_group("metadata","/")
            for key, value in self.metadata.items():
                f5h.create_array(ac,key, value)

    def get_data(self):
        """ 
        Python generator that saves the data passed by the source to a `*.h5` file and
        yields the data to the next object.
    
        Returns
        -------
        Dictionary containing a sample of the data set 
        {feature_name[key] : feature[values]}. 
        """
        
        self.writeflag = True
        f5h = self.get_initialized_file()      
        subf = self.get_filtered_features() 
        for data in self.source.get_data(): 
            if not self.writeflag: return     
            #create a group for each Sample
            ac = f5h.create_new_group(str(data['idx']))
            #store dict in the group
            if not subf:
                [f5h.create_array(ac,key, value) for key, value in data.items()]   
            else:
                [f5h.create_array(ac,key, value) for key, value in data.items() if key in subf]   
            yield data
            f5h.flush()
        self.add_metadata(f5h)
        f5h.close()
        


if TF_FLAG:


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
        """
        Class intended to write data from :class:`~acoupipe.pipeline.BasePipeline` 
        instances to a .tfrecord file that can be consumed by tensorflows 
        tf.data API. Stores data in binary format.
        """
        #: Name of the file to be saved. 
        name = File(filter=['*.tfrecords'], 
            desc="name of data file")   
        
        #: Dictionary with encoding functions (dict values) to convert data yielded by the pipeline to binary format of .tfrecord file.
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

        def _write_infofile(self, features):
            """internal function to write info file with additional data set information

            Parameters
            ----------
            features : dict
                features stored in the .tfrecord file
            """
            info_filename = ".".join(self.name.split(".")[:-1])
            with open(info_filename+".txt", 'w') as file:
                file.write("stored features \n")
                for featurename, feature in features.items():
                    encoder_func = self.encoder_funcs.get(featurename)
                    if encoder_func: # if it exists, the feature is stored
                        base_str = featurename + 2*"\t" + encoder_func.__name__ + 2*"\t" + f"type: {type(feature)}" + 2*"\t"
                        if encoder_func in (int_list_feature, float_list_feature):
                            if type(feature) is list:
                                file.write(
                                    base_str + 2*"\t" + f"length: {len(feature)}")
                            elif type(feature) is ndarray:
                                file.write(
                                    base_str + 2*"\t" + f"shape: {feature.shape}")        
                            else:
                                file.write(base_str)                       
                        else:
                            file.write(base_str)
                        file.write('\n')

        def save(self):
            """
            saves the output of the :meth:`get_data()` method of a :class:`~acoupipe.pipeline.BasePipeline` 
            derived class instance specified at :attr:`source` to .tfrecord file format.  
            """
            with tf.io.TFRecordWriter(self.name,options=self.options) as writer:
                for i,features in enumerate(self.source.get_data()):
                    encoded_features = {n:self.encoder_funcs[n](f) for (n,f) in features.items() if self.encoder_funcs.get(n)}
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                if self.write_infofile:
                    self._write_infofile(features)
                writer.close()
                
            
        def get_data(self):
            """ 
            Python generator that saves the data passed by the source to a `*.tfrecord` file and
            yields the data to the next object.
        
            Returns
            -------
            Dictionary containing a sample of the data set 
            {feature_name[key] : feature[values]}. 
            """
            with tf.io.TFRecordWriter(self.name,options=self.options) as writer:
                for i,features in enumerate(self.source.get_data()):
                    encoded_features = {n:self.encoder_funcs[n](f) for (n,f) in features.items() if self.encoder_funcs.get(n)}
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                    yield features
                if self.write_infofile:
                    self._write_infofile(features)
                writer.close()            
