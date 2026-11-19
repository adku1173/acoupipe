"""Base classes for generating microphone array datasets."""

import logging
from functools import partial

from acoupipe.datasets.base.legacy_config import ConfigBase
from acoupipe.datasets.features import Feature, combine_feature_funcs, get_tf_mappers, remove_unrequested
from acoupipe.pipeline import BasePipeline, DistributedPipeline, seed_key
from acoupipe.writer import WriteH5Dataset

from traits.api import Any, Dict, HasPrivateTraits, Instance, Int, Property


def _set_pipeline_seed(pipeline, size, split, seed):
    """Set the number of samples and the seed of the pipeline for a dataset split.

    The samplers of the data sample with index ``idx`` are seeded from
    ``(seed, seed_key(split), idx, sampler key)`` (see :func:`~acoupipe.pipeline.reseed_sampler`).
    Different splits and dataset seeds therefore yield independent random streams for any number
    of samples.

    Parameters
    ----------
    pipeline : instance of class BasePipeline
        the pipeline object holding the sampler classes
    size : int
        number of samples to be yielded by the pipeline
    split : str
        name of the dataset split, e.g. 'training', 'validation' or 'test'
    seed : int
        non-negative seed of the dataset
    """
    pipeline.numsamples = size
    pipeline.seed = (seed, seed_key(split))


class DatasetBase(HasPrivateTraits):
    """
    Base class for generating microphone array datasets with specified features and labels.

    Attributes
    ----------
    config : ConfigBase
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

    config = Any(desc='configuration object')
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
        if config is None:
            config = ConfigBase()
        if not hasattr(config, 'get_sampler'):
            msg = 'config must provide get_sampler()'
            raise TypeError(msg)
        self.config = config
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

    def get_features(self, features, f, num):
        """
        Get the combined feature function and the feature objects of the dataset.

        If the config provides its own ``get_features`` method (e.g.
        :class:`~acoupipe.datasets.base.config.Config`), the call is delegated to
        it. Otherwise, the feature functions are called in the following order: the prepare
        function of the config (if any), the default features requested by name,
        the custom feature objects (Feature instances) and finally a cleanup that
        removes all outputs that were not requested (see
        :func:`~acoupipe.datasets.features.remove_unrequested`).

        Parameters
        ----------
        features : list
            List of features included in the dataset. Contains names of default
            features (str) and/or custom feature objects (Feature).
        f : float
            The center frequency or list of frequencies of the dataset.
            If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        feature_func : callable
            Single combined callable with signature ``f(sampler) -> dict`` (see
            :func:`~acoupipe.datasets.features.combine_feature_funcs`). It chains the
            prepare function, the feature functions of all requested features and the
            cleanup, and returns their merged outputs as one dictionary.
        feature_objects : list of Feature
            Feature objects in the order in which they are calculated.
        """
        if hasattr(self.config, 'get_features'):
            return self.config.get_features(features, f, num)
        feature_funcs = []
        prepare_func = self.config.get_prepare_func() if hasattr(self.config, 'get_prepare_func') else None
        if prepare_func is not None:
            feature_funcs.append(prepare_func)
        default_feature_names = [feat for feat in features if isinstance(feat, str)]
        default_features = []
        if hasattr(self.config, 'get_default_features'):
            default_features = self.config.get_default_features(default_feature_names, f, num)
        custom_features = [feat for feat in features if isinstance(feat, Feature)]
        feature_objects = default_features + custom_features
        feature_funcs += [feature.get_feature_func() for feature in feature_objects]
        keep = default_feature_names + [feature.name for feature in custom_features]
        feature_funcs.append(partial(remove_unrequested, keep=keep))
        return combine_feature_funcs(feature_funcs), feature_objects

    def generate(self, features, size, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Generate dataset samples iteratively.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        size : int
            Size of the dataset (number of source cases).
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================
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
        Generate features iteratively (example below requires a dataset configuration).

        .. code-block:: python

            from acoupipe.datasets.synthetic import DatasetSynthetic

            # define the features
            features = ['csm', 'source_strength_analytic', 'loc']
            f = 1000
            num = 3

            # generate the dataset
            generator = DatasetSynthetic().generate(f=f, num=num, split='training', size=2, features=features)

            # iterate over the dataset
            for data in generator:
                print(data)
        """
        pipeline = self.get_pipeline_instance()
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.get_features(features, f, num)
        _set_pipeline_seed(pipeline, size, split, self.seed)
        yield from pipeline.get_data(progress_bar=progress_bar, start_idx=start_idx)

    def get_ray_dataset(self, features, size, split='training', f=None, num=0, start_idx=0):
        """Return the dataset samples as a lazy :class:`ray.data.Dataset`.

        The samples are calculated by :attr:`tasks` Ray Data actors, each calculating up to
        :attr:`samples_per_task` samples per call. The dataset can be consumed directly, e.g. with
        ``iter_torch_batches()``, or written to file formats supported by Ray Data. Complex
        features are stored as two real columns ``<name>_real`` and ``<name>_imag``. Requires
        Ray Data (``ray[data]``).

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        size : int
            Size of the dataset (number of source cases).
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line), see :meth:`generate`.
        start_idx : int, optional
            Starting sample index (default is 0).

        Returns
        -------
        ray.data.Dataset
            Dataset with one row per sample and the features and "idx" as columns.

        Examples
        --------
        Write CSMs to Parquet files (example requires a dataset configuration).

        .. code-block:: python

            from acoupipe.datasets.synthetic import DatasetSynthetic

            dataset = DatasetSynthetic(tasks=4).get_ray_dataset(features=['csm'], f=1000, size=100)
            dataset.write_parquet('/tmp/csm')  # columns csm_real, csm_imag, idx
        """
        from acoupipe.raydata.pipeline import RayDataPipeline  # noqa: PLC0415

        pipeline = RayDataPipeline(
            numworkers=self.tasks, samples_per_task=self.samples_per_task, remote_args=self.remote_args
        )
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.get_features(features, f, num)
        _set_pipeline_seed(pipeline, size, split, self.seed)
        return pipeline.get_dataset(start_idx=start_idx)

    def save_h5(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
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
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================
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

            from acoupipe.datasets.synthetic import DatasetSynthetic

            # define the features
            features = ['csm', 'source_strength_analytic', 'loc']
            f = 1000
            num = 3

            # save the dataset
            dataset = DatasetSynthetic().save_h5(f=f, num=num, split='training', size=10, features=features, name='/tmp/example.h5')
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, _ = self.get_features(features, f, num)
        _set_pipeline_seed(pipeline, size, split, self.seed)
        WriteH5Dataset(
            name=name,
            source=pipeline,
        ).save(progress_bar, start_idx)  # start the calculation

    def save_tfrecord(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
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
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================
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

            from acoupipe.datasets.synthetic import DatasetSynthetic

            # define the features
            features = ['csm', 'source_strength_analytic', 'loc']
            f = 1000
            num = 3

            # save the dataset
            dataset = DatasetSynthetic().save_tfrecord(
                f=f, num=num, split='training', size=10, features=features, name='/tmp/example.tfrecord'
            )
        """
        from acoupipe.tf.writer import WriteTFRecord  # noqa: PLC0415

        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.feature_func, feature_objects = self.get_features(features, f, num)
        _set_pipeline_seed(pipeline, size, split, self.seed)
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

    def get_output_signature(self, features, f=None, num=0):
        """Get the output signature of the dataset.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        dict
            Output signature of the dataset.
        """
        import tensorflow as tf  # noqa: PLC0415

        signature = {}
        _, feature_objects = self.get_features(features, f, num)
        _, shapes, dtypes = get_tf_mappers(feature_objects)
        for feature in features:
            signature[feature] = tf.TensorSpec(shapes[feature], dtypes[feature])
        return signature

    def get_tf_dataset(self, features, size, split='training', f=None, num=0, start_idx=0, progress_bar=False):
        """Get a TensorFlow dataset from the generated data.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The feature "idx" is always included.
        size : int
            Size of the dataset (number of source cases).
        split : str
            Split name for the dataset ('training', 'validation' or 'test'). Defaults to 'training'.
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================
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
        pipeline.feature_func, _ = self.get_features(features, f, num)
        _set_pipeline_seed(pipeline, size, split, self.seed)
        features = features + ['idx']
        output_signature = self.get_output_signature(features, f=f, num=num)
        return tf.data.Dataset.from_generator(
            partial(self._generate, pipeline=pipeline, start_idx=start_idx, progress_bar=progress_bar),
            output_signature=output_signature,
        )

    def get_tfrecord_parser(self, features, f, num):
        """Get a parser function for a TFRecord dataset.

        The parser function can be used to parse the TFRecord dataset into a TensorFlow dataset.
        Complex-valued features of the dataset are encoded as two real-valued float32 features (real and imaginary part) stacked at
        the least axis of the array. The parser function decodes the features back to complex-valued features.
        It can be used as follows:

        Parameters
        ----------
        features : list
            List of features included in the dataset.
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        function
            A parser function that can be used to parse the TFRecord dataset.

        Examples
        --------
        (Example requires proper dataset and file paths)

        .. code-block:: python

            from acoupipe.datasets.synthetic import DatasetSynthetic

            # define the features
            features = ['csm', 'source_strength_analytic', 'loc']
            f = 1000
            num = 3

            # save the dataset
            dataset = DatasetSynthetic().save_tfrecord(
                f=f, num=num, split='training', size=10, features=features, name='/tmp/example.tfrecord'
            )

            # parse the dataset
            parser = dataset.get_tfrecord_parser(features, f, num)
            dataset = tf.data.TFRecordDataset('/tmp/example.tfrecord')
            dataset = iter(dataset.map(parser))
            data = next(dataset)

        """
        from acoupipe.tf.encodings import complex_list_feature  # noqa: PLC0415

        import tensorflow as tf  # noqa: PLC0415

        _, feature_objects = self.get_features(features, f, num)
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
