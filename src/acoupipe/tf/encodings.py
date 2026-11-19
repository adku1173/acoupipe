"""Encode features for TFRecord files and infer their TensorFlow dtypes and shapes."""

import numpy as np
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
