"""Write data to the TensorFlow TFRecord format."""

from acoupipe.tf.encodings import int_list_feature
from acoupipe.writer import BaseWriteDataset

import numpy as np
import tensorflow as tf
from traits.api import Callable, Dict, File, List, Str, Trait


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
