"""Base classes for generating microphone array datasets."""

import logging
from functools import partial

from acoupipe.datasets._shared.utils import set_pipeline_seed
from acoupipe.datasets.base.config import Config
from acoupipe.datasets.features import get_tf_mappers
from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.writer import WriteH5Dataset

from traits.api import Dict, HasPrivateTraits, Instance, Int, Property


class Dataset(HasPrivateTraits):
    """
    Generate microphone array datasets from a :class:`~acoupipe.datasets.base.config.Config`.

    Frequency settings are part of the configuration: set ``config.parameters.f``
    and ``config.parameters.num`` instead of passing ``f`` and ``num`` to the
    generation methods. Legacy configurations based on
    :class:`~acoupipe.datasets.base.legacy_config.ConfigBase` are used with
    :class:`~acoupipe.datasets.base.legacy_dataset.DatasetBase`.

    Attributes
    ----------
    config : Config
        Configuration object for dataset generation.
    tasks : int
        Number of parallel tasks for data generation. Defaults to 1 (sequential calculation).
        For tasks > 1, limit each worker process to a single thread, see :ref:`parallel-generation`.
    samples_per_task : int
        Maximum number of samples calculated per remote call if tasks > 1. Defaults to 16.
    seed : int
        Non-negative seed of the dataset. Defaults to 0. A sample is determined by the seed, the split
        name and its index; different seeds or split names yield independent samples.
    """

    config = Instance(Config, desc='configuration object')
    tasks = Property(desc='number of parallel tasks for data generation')
    remote_args = Dict({})
    samples_per_task = Int(16, desc='maximum number of samples calculated per remote call')
    seed = Int(0, desc='non-negative seed of the dataset')
    #: logger instance to log calculation times for each data sample
    logger = Property(desc='Logger instance to log timing statistics')

    # private
    _logger = Instance(logging.Logger, desc='Internal logger instance')
    _tasks = Int(1, desc='number of parallel tasks for data generation')

    def __init__(self, config=None, tasks=1, remote_args=None, logger=None, samples_per_task=16, seed=0):
        HasPrivateTraits.__init__(self)
        self.tasks = tasks
        self.samples_per_task = samples_per_task
        self.seed = seed
        self.config = Config() if config is None else config
        self.remote_args = remote_args or {}
        self.logger = logger

    def _get_logger(self):
        if self._logger is None:
            self._logger = self._get_default_logger()
        return self._logger

    def _set_logger(self, logger):
        self._logger = logger

    def _get_default_logger(self):
        """Set up standard logging to stdout, stderr."""
        logger = logging.getLogger(__name__)
        logger.propagate = False  # don't propagate to the root logger!
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter('%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d %(message)s', datefmt='%H:%M:%S'),
        )
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stream_handler)
        return logger

    def _get_tasks(self):
        return self._tasks

    def _set_tasks(self, tasks):
        self._tasks = tasks

    def get_pipeline_instance(self):
        if self.tasks > 1:
            return DistributedPipeline(
                numworkers=self.tasks, samples_per_task=self.samples_per_task, remote_args=self.remote_args
            )
        return BasePipeline()

    def _generate(self, pipeline, progress_bar, start_idx):
        """Generate dataset samples.

        Parameters
        ----------
        pipeline : BasePipeline
            Pipeline object.
        progress_bar : bool, optional
            Whether to show a progress bar.
        start_idx : int, optional
            Starting sample index.

        Returns
        -------
        data
            Dataset samples.
        """
        yield from pipeline.get_data(progress_bar=progress_bar, start_idx=start_idx)

    def generate(self, features, size, split='training', start_idx=0, progress_bar=True):
        """Generate dataset samples iteratively.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        size : int
            Size of the dataset (number of source cases).
        start_idx : int, optional
            Starting sample index (default is 0).
        progress_bar : bool, optional
            Whether to show a progress bar (default is True).

        Yields
        ------
        data : dict
            Generator that yields dataset samples as dictionaries containing the feature names as keys.

        Examples
        --------
        Generate features iteratively.

        .. code-block:: python

            from acoupipe.datasets import Dataset, SyntheticConfig

            config = SyntheticConfig()
            config.parameters.add_parameter('f', 1000.0)
            generator = Dataset(config=config).generate(features=['c', 'f'], split='training', size=2)

            # iterate over the dataset
            for data in generator:
                print(data)
        """
        pipeline = self.get_pipeline_instance()
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.config.get_features(features)
        set_pipeline_seed(pipeline, size, split, self.seed)
        yield from pipeline.get_data(progress_bar=progress_bar, start_idx=start_idx)

    def save_h5(self, features, size, name, split='training', start_idx=0, progress_bar=True):
        """Save dataset to a HDF5 file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        size : int
            Size of the dataset (number of source cases).
        name : str
            Name of the HDF5 file.
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        start_idx : int, optional
            Starting sample index (default is 0).
        progress_bar : bool, optional
            Whether to show a progress bar (default is True).

        Returns
        -------
        None

        Examples
        --------
        Save features to a HDF5 file (example requires proper file path).

        .. code-block:: python

            from acoupipe.datasets import Dataset, SyntheticConfig

            Dataset(config=SyntheticConfig()).save_h5(features=['c', 'fs'], split='training', size=10, name='/tmp/example.h5')
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.config.get_features(features)
        set_pipeline_seed(pipeline, size, split, self.seed)
        WriteH5Dataset(
            name=name,
            source=pipeline,
        ).save(progress_bar, start_idx)  # start the calculation

    def save_tfrecord(self, features, size, name, split='training', start_idx=0, progress_bar=True):
        """Save dataset to a .tfrecord file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        size : int
            Size of the dataset (number of source cases).
        name : str
            Name of the TFRecord file.
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        start_idx : int, optional
            Starting sample index (default is 0).
        progress_bar : bool, optional
            Whether to show a progress bar (default is True).

        Returns
        -------
        None

        Examples
        --------
        Save features to a TFRecord file (example requires proper file path).

        .. code-block:: python

            import numpy as np
            from acoupipe.datasets import Dataset, SyntheticConfig

            # TFRecord files require dtype and shape of each feature
            config = SyntheticConfig()
            config.feature('c_squared', lambda parameters: parameters.c**2, dtype=np.float64, shape=())
            Dataset(config=config).save_tfrecord(features=['c_squared'], split='training', size=10, name='/tmp/example.tfrecord')
        """
        from acoupipe.tf.writer import WriteTFRecord  # noqa: PLC0415

        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, feature_objects = self.config.get_features(features)
        set_pipeline_seed(pipeline, size, split, self.seed)
        encoders, shapes, _ = get_tf_mappers(feature_objects)
        # get features with varying length to handle them correctly in the TFRecord writer
        shape_features = []
        for feature, shape in shapes.items():
            # if more then one None in shape, we have a varying length feature
            if list(shape).count(None) > 1:
                shape_features.append(feature)

        WriteTFRecord(
            name=name,
            source=pipeline,
            shape_features=shape_features,
            encoder_funcs=encoders,
        ).save(
            progress_bar,
            start_idx,
        )

    def get_output_signature(self, features):
        """Get the output signature of the dataset.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.

        Returns
        -------
        dict
            Output signature of the dataset.
        """
        import tensorflow as tf  # noqa: PLC0415

        signature = {}
        _, feature_objects = self.config.get_features(features)
        _, shapes, dtypes = get_tf_mappers(feature_objects)
        for feature in features:
            signature[feature] = tf.TensorSpec(shapes[feature], dtypes[feature])
        return signature

    def get_tf_dataset(self, features, size, split='training', start_idx=0, progress_bar=False):
        """Get a TensorFlow dataset from the generated data.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        size : int
            Size of the dataset (number of source cases).
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        start_idx : int, optional
            Starting sample index (default is 0).
        progress_bar : bool, optional
            Whether to show a progress bar (default is False).

        Returns
        -------
        tf.data.Dataset
            TensorFlow dataset containing the generated data. The dataset elements have the structure defined
            by the output_signature, which is based on the shapes of dataset features.
        """
        import tensorflow as tf  # noqa: PLC0415

        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.config.get_features(features)
        set_pipeline_seed(pipeline, size, split, self.seed)
        features = features + ['idx']
        output_signature = self.get_output_signature(features)
        return tf.data.Dataset.from_generator(
            partial(self._generate, pipeline=pipeline, start_idx=start_idx, progress_bar=progress_bar),
            output_signature=output_signature,
        )

    def get_tfrecord_parser(self, features):
        """Get a parser function for a TFRecord dataset.

        The parser function can be used to parse the TFRecord dataset into a TensorFlow dataset.
        Complex-valued features of the dataset are encoded as two real-valued float32 features (real and imaginary part) stacked at
        the least axis of the array. The parser function decodes the features back to complex-valued features.
        It can be used as follows:

        Parameters
        ----------
        features : list
            List of features included in the dataset.

        Returns
        -------
        function
            A parser function that can be used to parse the TFRecord dataset.

        Examples
        --------
        (Example requires proper dataset and file paths)

        .. code-block:: python

            import numpy as np
            import tensorflow as tf
            from acoupipe.datasets import Dataset, SyntheticConfig

            features = ['c_squared']
            config = SyntheticConfig()
            config.feature('c_squared', lambda parameters: parameters.c**2, dtype=np.float64, shape=())
            dataset = Dataset(config=config)
            dataset.save_tfrecord(features=features, split='training', size=10, name='/tmp/example.tfrecord')

            # parse the dataset
            parser = dataset.get_tfrecord_parser(features)
            tf_dataset = tf.data.TFRecordDataset('/tmp/example.tfrecord')
            data = next(iter(tf_dataset.map(parser)))
        """
        from acoupipe.tf.encodings import complex_list_feature  # noqa: PLC0415

        import tensorflow as tf  # noqa: PLC0415

        _, feature_objects = self.config.get_features(features)
        encoders, shapes, dtypes = get_tf_mappers(feature_objects)
        features = features + ['idx']

        feature_description = {}

        for feature in features:
            dtype = dtypes[feature]
            if dtype in [tf.complex64, tf.complex128]:  # complex not supported for tfrecord files
                shapes[feature] = shapes[feature] + (2,)
                dtype = tf.float32 if dtype == tf.complex64 else tf.float64
            shape = shapes[feature]
            if None in shape:
                feature_description[feature] = tf.io.VarLenFeature(dtype)
                if list(shape).count(None) > 1:
                    shape_key = f'{feature}_shape'
                    feature_description[shape_key] = tf.io.FixedLenFeature((len(shape),), tf.int64)
            else:
                feature_description[feature] = tf.io.FixedLenFeature(shape, dtype)

        def _parse_function(example_proto):
            data = tf.io.parse_single_example(example_proto, feature_description)
            for feature in features:
                value = data[feature]
                shape = shapes[feature]
                shape_key = f'{feature}_shape'
                shape_tensor = data.get(shape_key)

                # Only densify sparse tensors
                if isinstance(value, tf.SparseTensor):
                    value = tf.sparse.to_dense(value)
                    if shape_tensor is not None:
                        value = tf.reshape(value, tf.cast(shape_tensor, tf.int32))
                    elif None in shape:
                        target_shape = [s if s is not None else -1 for s in shape]
                        value = tf.reshape(value, target_shape)
                elif shape_tensor is not None:
                    # Dynamic shape provided but value already dense
                    value = tf.reshape(value, tf.cast(shape_tensor, tf.int32))

                encoder = encoders[feature]
                if encoder is complex_list_feature:
                    value = tf.complex(value[..., 0], value[..., 1])

                data[feature] = value
            return data

        return _parse_function


__all__ = ['Dataset']
