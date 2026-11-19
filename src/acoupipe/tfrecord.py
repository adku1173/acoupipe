"""Write data to the TensorFlow TFRecord format.

This module requires TensorFlow. It is only imported when TFRecord files or TensorFlow datasets are
used, so that importing other AcouPipe modules, e.g. in Ray workers, does not load TensorFlow.
"""

from acoupipe.writer import BaseWriteDataset

import numpy as np
import tensorflow as tf
from traits.api import Callable, Dict, File, List, Str, Trait


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
    """Decide which encoder function, TF dtype and TF shape to use for a feature.

    The feature is given by its numpy / Python dtype and shape.

    Returns
    -------
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
    For features whose shapes may vary (contain ``None``), list their names
    in :attr:`shape_features`
    to have the runtime shape stored as an auxiliary ``<name>_shape`` int64 feature.
    """

    #: Name of the file to be saved.
    name = File(filter=['*.tfrecords'], desc='name of data file')

    #: Dictionary with encoding functions (dict values) to convert data yielded
    #: by the pipeline to binary .tfrecord format.
    #: The key values of this dictionary are the feature names of the
    #: samples yielded by the :attr:`source` object.
    encoder_funcs = Dict(
        key_trait=Str(),
        value_trait=Callable(),
        desc='encoding functions to convert data yielded by the pipeline to binary TFRecord format',
    )

    #: Trait to set specific options to the .tfrecord file.
    options = Trait(None, tf.io.TFRecordOptions)

    #: List of feature names for which the shape should be stored alongside
    #: the data (as ``<name>_shape``).
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
