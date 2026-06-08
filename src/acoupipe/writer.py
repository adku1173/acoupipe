"""Provides classes to store the data extracted by :class:`~acoupipe.pipeline.BasePipeline` derived classes.

Purpose of the Writer Module
----------------------------
The :code:`writer.py` module provides classes to store the data extracted by the pipeline.
The current implementation includes classes to save data in a container-like file format (.h5 file with the :code:`WriteH5Dataset` class) or
binary format (.tfrecord file with the :code:`WriteTFRecord` class).
The latter can be efficiently consumed by the Tensorflow framework for machine learning.

.. code-block:: python

    file_writer = acoupipe.writer.WriteH5Dataset(
        source=pipeline,
    )

    file_writer.save()


"""

from datetime import datetime
from os import path

import numpy as np
from acoular import config
from h5py import File as H5File
from traits.api import Bool, Callable, Dict, File, Instance, List, Str, Trait

from acoupipe.config import TF_FLAG
from acoupipe.pipeline import DataGenerator


class BaseWriteDataset(DataGenerator):
    """Base class intended to write data from :class:`~acoupipe.pipeline.BasePipeline` instances to a specific file format.

    This class has no functionality and should not be used.
    """

    #: source instance that has to be of type :class:`~acoupipe.pipeline.DataGenerator`
    source = Instance(DataGenerator)

    def save(self):
        """Save data from a :class:`~acoupipe.pipeline.BasePipeline` instance specified at :attr:`source` to file."""
        # write to File...

    def get_data(self, progress_bar=True, start_idx=1):
        """Python generator that saves source output data to file and passes the data to the next object.

        Parameters
        ----------
        progress_bar : bool
            If True, a progress bar is shown.
        start_idx : int
            Index of the first sample which is written to file.

        Returns
        -------
        dict
            Dictionary containing a sample of the data set
            {feature_name[key] : feature[values]}.
        """
        yield from self.source.get_data(progress_bar, start_idx)


class WriteH5Dataset(BaseWriteDataset):
    """Class intended to write data to a `.h5` file."""

    #: Name of the file to be saved.
    name = File(filter=['*.h5'], desc='name of data file')

    # #: Number of samples to write to file by :meth:`result` method.
    # #: defaults to -1 (write as long as source yields data).
    # numsamples_write = Int(-1)

    #: flag that can be raised to stop file writing when running in detached thread
    writeflag = Bool(True)

    #: a list with names of the features to be saved. By default [],
    #: meaning that all features comming from the source will be saved.
    features = List([], desc='the names of the features to be saved')

    #: Metadata to be stored in HDF5 file object
    metadata = Dict(desc='metadata to be stored in .h5 file')

    def create_filename(self):
        if self.name == '':
            name = datetime.now().isoformat('_').replace(':', '-').replace('.', '_')  # noqa: DTZ005
            self.name = path.join(config.td_dir, name + '.h5')

    def get_file(self):
        self.create_filename()
        return H5File(self.name, mode='w')

    def get_filtered_features(self):
        if self.features:
            return self.features.copy() + ['idx'] if 'idx' not in self.features else self.features
        return None

    def _add_data(self, f5h, data, subf):
        dataset_idx = str(data['idx'])
        # create a group for each Sample
        f5h.create_group(dataset_idx)
        # store dict in the group
        if not subf:
            [f5h.create_dataset(f'{dataset_idx}/{key}', data=value) for key, value in data.items()]
        else:
            [f5h.create_dataset(f'{dataset_idx}/{key}', data=value) for key, value in data.items() if key in subf]

    def _add_metadata(self, f5h):
        """Add metadata to .h5 file."""
        nitems = len(self.metadata.items())
        if nitems > 0:
            f5h.create_group('metadata')
            for key, value in self.metadata.items():
                f5h.create_dataset(f'metadata/{key}', data=value)

    def save(self, progress_bar=True, start_idx=1):
        """Save the output of the :meth:`get_data()` method of :class:`~acoupipe.pipeline.BasePipeline` to .h5 file format."""
        f5h = self.get_file()
        subf = self.get_filtered_features()
        for data in self.source.get_data(progress_bar, start_idx):
            self._add_data(f5h, data, subf)
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()

    def get_data(self, progress_bar=True, start_idx=1):
        """Python generator that saves the data passed by the source to a `*.h5` file and yields the data to the next object.

        Returns
        -------
        dict
            Dictionary containing a sample of the data set
            {feature_name[key] : feature[values]}.
        """
        self.writeflag = True
        f5h = self.get_file()
        subf = self.get_filtered_features()
        for data in self.source.get_data(progress_bar, start_idx):
            if not self.writeflag:
                return
            self._add_data(f5h, data, subf)
            yield data
            f5h.flush()
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()


if TF_FLAG:
    import tensorflow as tf

    def bytes_feature(value):
        """bytes_list from a bytes or str or array-like of those."""
        if tf.is_tensor(value):
            value = value.numpy()

        arr = np.asarray(value)

        # Unicode -> bytes
        if arr.dtype.kind == 'U':
            arr = arr.astype('S')

        # Flatten and ensure bytes
        flat = arr.reshape(-1)
        flat = [bytes(x) for x in flat]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=flat))

    def int_list_feature(value):
        """int64_list from scalar or array-like of ints/bools."""
        if tf.is_tensor(value):
            value = value.numpy()
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))

    def float_list_feature(value):
        """float_list from scalar or array-like of floats."""
        if tf.is_tensor(value):
            value = value.numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return tf.train.Feature(float_list=tf.train.FloatList(value=arr))

    def complex_list_feature(value):
        """float_list from scalar or array-like of complex values.

        Encodes complex z as [Re(z), Im(z)] along a last axis of length 2,
        then flattens to 1D float32.
        """
        if tf.is_tensor(value):
            value = value.numpy()
        value = np.asarray(value)
        re = np.real(value)[..., np.newaxis]
        im = np.imag(value)[..., np.newaxis]
        stacked = np.concatenate([re, im], axis=-1)  # shape (..., 2)
        flat = stacked.astype(np.float32).reshape(-1)
        return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

    def infer_tf_encoding(dtype, shape):
        """
        Decide which encoder function, TF dtype and TF shape to use
        for a feature with given numpy / Python dtype and shape.

        Returns:
            encoder_func, tf_dtype, tf_shape
        """
        tf_dtype = tf.dtypes.as_dtype(dtype)

        # Complex -> tf.complex dtype
        if tf_dtype.is_complex:
            encoder = complex_list_feature
            out_tf_dtype = tf_dtype
            out_shape = tuple(shape) if shape is not None else ()
            return encoder, out_tf_dtype, out_shape

        # Floats -> float32 list
        if tf_dtype.is_floating:
            encoder = float_list_feature
            out_tf_dtype = tf.float32
            out_shape = tuple(shape) if shape is not None else ()
            return encoder, out_tf_dtype, out_shape

        # Integers / bool -> int64 list
        if tf_dtype.is_integer or tf_dtype == tf.bool:
            encoder = int_list_feature
            out_tf_dtype = tf.int64
            out_shape = tuple(shape) if shape is not None else ()
            return encoder, out_tf_dtype, out_shape

        # Strings -> bytes
        if tf_dtype == tf.string:
            encoder = bytes_feature
            out_tf_dtype = tf.string
            out_shape = tuple(shape) if shape is not None else ()
            return encoder, out_tf_dtype, out_shape
        msg = f'Unsupported dtype {dtype!r} (tf: {tf_dtype})'
        raise TypeError(msg)

    class WriteTFRecord(BaseWriteDataset):
        """Write pipeline output to TFRecord.

        Serializes samples from a :class:`~acoupipe.pipeline.BasePipeline` into the TensorFlow
        TFRecord format, using encoder functions provided via :attr:`encoder_funcs`.
        For features whose shapes may vary (contain ``None``), list their names in :attr:`shape_features`
        to have the runtime shape stored as an auxiliary ``<name>_shape`` int64 feature.
        """

        #: Name of the file to be saved.
        name = File(filter=['*.tfrecords'], desc='name of data file')

        #: Dictionary with encoding functions (dict values) to convert data yielded by the pipeline to binary .tfrecord format.
        #: The key values of this dictionary are the feature names specified in the :attr:`features` attribute
        #: of the :attr:`source` object.
        encoder_funcs = Dict(
            key_trait=Str(),
            value_trait=Callable(),
            desc='encoding functions to convert data yielded by the pipeline to binary TFRecord format',
        )

        #: Trait to set specific options to the .tfrecord file.
        options = Trait(None, tf.io.TFRecordOptions)

        #: List of feature names for which the shape should be stored alongside the data (as ``<name>_shape``).
        shape_features = List(Str, desc='features whose shapes are written along with the data')

        def _encode_sample(self, features, encoders):
            sample = dict(features)
            for name in self.shape_features:
                shape_key = f'{name}_shape'
                if name in sample and shape_key not in sample:
                    shape = sample[name].shape
                    if not np.isrealobj(sample[name]):
                        shape = shape + (2,)
                    sample[shape_key] = np.array(shape, dtype=np.int64)
                    encoders.setdefault(shape_key, int_list_feature)

            return {n: encoders[n](f) for (n, f) in sample.items() if encoders.get(n)}

        def save(self, progress_bar=True, start_idx=1):
            """
            Save pipeline output to TFRecord.

            Parameters
            ----------
            progress_bar : bool, optional
                Whether to display a progress bar while writing, by default True.
            start_idx : int, optional
                Starting sample index (used to seed the pipeline), by default 1.
            """
            encoders = dict(self.encoder_funcs)
            with tf.io.TFRecordWriter(self.name, options=self.options) as writer:
                for _i, features in enumerate(self.source.get_data(progress_bar, start_idx)):
                    encoded_features = self._encode_sample(features, encoders)
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())

        def get_data(self, progress_bar=True, start_idx=1):
            """
            Stream pipeline output to TFRecord and yield samples.

            Parameters
            ----------
            progress_bar : bool, optional
                Whether to display a progress bar while writing, by default True.
            start_idx : int, optional
                Starting sample index (used to seed the pipeline), by default 1.

            Yields
            ------
            dict
                One sample of the dataset as a mapping of feature names to values
                (original features only; shape metadata is written but not yielded).
            """
            encoders = dict(self.encoder_funcs)
            with tf.io.TFRecordWriter(self.name, options=self.options) as writer:
                for _i, features in enumerate(self.source.get_data(progress_bar, start_idx)):
                    encoded_features = self._encode_sample(features, encoders)
                    example = tf.train.Example(features=tf.train.Features(feature=encoded_features))
                    # Serialize to string and write on the file
                    writer.write(example.SerializeToString())
                    yield features
                    writer.flush()
                writer.close()
