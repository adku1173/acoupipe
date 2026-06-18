"""Base classes for generating microphone array datasets."""

import logging
from functools import partial

from traits.api import Any, Callable, Dict, HasPrivateTraits, Instance, Int, List, Property

from acoupipe.base import BaseSampler
from acoupipe.config import TF_FLAG
from acoupipe.datasets.callbacks import call_with_supported_context
from acoupipe.datasets.features import BaseFeatureCatalog, BaseFeatureCollectionBuilder
from acoupipe.datasets.parameters import ParameterSet
from acoupipe.datasets.utils import set_pipeline_seeds
from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.sampler import AttributeSampler
from acoupipe.writer import WriteH5Dataset

if TF_FLAG:
    import tensorflow as tf

    from acoupipe.writer import WriteTFRecord, complex_list_feature


class Config(HasPrivateTraits):
    """Configuration class for generating microphone array datasets.

    Configs define the simulation runtime and own a ``parameters`` object that
    holds the current Monte-Carlo state. The Dataset pipeline may use samplers
    internally to mutate those parameters, but simulation code should consume
    the parameter object rather than sampler dictionary entries.

    Notes
    -----
    ``_sampler_key_limit`` is an internal guard for the sampler key namespace.
    Dataset prepare functions use fixed integer keys to look up samplers with
    specific meanings. Some of those samplers are optional, but their keys must
    remain reserved so that appended parameter samplers cannot accidentally
    occupy a key that a prepare function interprets differently. Subclasses
    should set this to the highest reserved sampler key used by their private
    sampler/prepare-function contract. It is private infrastructure and not
    part of the user-facing Dataset API.
    """

    _sampler_key_limit = Int(-1, desc='highest internally reserved sampler key')
    parameters = Any(desc='Monte-Carlo parameters')
    _parameter_samplers = List(Instance(BaseSampler), desc='samplers created for parameter paths')
    _registered_prepare_funcs = List(Callable, desc='user-registered prepare callbacks')
    _registered_feature_funcs = List(Callable, desc='user-registered feature callbacks')
    _registered_feature_names = List(desc='names of user-registered feature callbacks')
    _registered_feature_metadata = Dict(desc='dtype and shape metadata for user-registered feature callbacks')

    def __init__(self, **traits):
        parameters = traits.pop('parameters', None)
        if parameters is None:
            parameters = self.create_parameters()
        super().__init__(parameters=parameters, **traits)

    def create_parameters(self):
        """Create the Monte-Carlo parameter object owned by this config."""
        return ParameterSet()

    def _get_legacy_sampler(self):
        """Return transitional samplers not yet backed by ``parameters``.

        This private hook exists while older Dataset configs still keep some
        Monte-Carlo state in sampler/helper objects. Future refactors should
        move those values into ``parameters`` and make this hook unnecessary.
        Older external configs may still define ``get_sampler()``; this method
        calls only those subclass implementations as a compatibility bridge.

        Implementation detail
        ---------------------
        The lookup walks ``type(self).__mro__`` and inspects each subclass
        ``__dict__`` until reaching ``Config``. This intentionally skips
        ``Config.get_sampler`` itself. If we resolved methods through normal
        attribute lookup (or included ``Config``), ``_get_legacy_sampler``
        could call ``Config.get_sampler`` again, which calls this method and
        causes recursion. Using ``__mro__`` + direct ``__dict__`` access keeps
        legacy subclass overrides working while avoiding that loop.
        """
        for cls in type(self).__mro__:
            if cls is Config:
                break
            legacy_get_sampler = cls.__dict__.get('get_sampler')
            if legacy_get_sampler is not None:
                return legacy_get_sampler(self)
        return {}

    def _resolve_parameter_path(self, path):
        """Return the value at a direct or dotted parameter path."""
        if not isinstance(path, str) or not path or any(part == '' for part in path.split('.')):
            msg = f'Unsupported sampling path "{path}". The parameter object has no such attribute.'
            raise ValueError(msg)
        value = self.parameters
        for part in path.split('.'):
            if not hasattr(value, part):
                msg = f'Unsupported sampling path "{path}". The parameter object has no such attribute.'
                raise ValueError(msg)
            value = getattr(value, part)
        return value

    def sample(self, path, random_var=None, random_func=None, sampler_class=None, equal_value=True, **sampler_kwargs):
        """Register a sampler for a direct or dotted parameter attribute."""
        self._resolve_parameter_path(path)
        if sampler_class not in (None, AttributeSampler):
            msg = f'Sampling path "{path}" supports only AttributeSampler.'
            raise ValueError(msg)
        if random_var is None and random_func is None:
            msg = f'Sampling path "{path}" requires random_var or random_func.'
            raise ValueError(msg)
        sampler = AttributeSampler(
            target=self.parameters,
            attribute=path,
            parameters=self.parameters,
            random_var=random_var,
            random_func=random_func,
            equal_value=equal_value,
            **sampler_kwargs,
        )
        self._parameter_samplers.append(sampler)
        return sampler

    def prepare(self, func):
        """Register a callback that runs after sampling and before feature extraction."""

        def prepare_func(sampler, data):
            result = call_with_supported_context(
                func,
                parameters=self.parameters,
                sampler=sampler,
                data=data,
            )
            return result if isinstance(result, dict) else {}

        self._registered_prepare_funcs.append(prepare_func)
        return func

    def feature(self, name, func, dtype=None, shape=None):
        """Register a named feature callback for Dataset generation."""
        if (dtype is None) != (shape is None):
            msg = f'Feature "{name}" metadata requires both dtype and shape.'
            raise ValueError(msg)

        def feature_func(sampler, data):
            value = call_with_supported_context(
                func,
                parameters=self.parameters,
                sampler=sampler,
                data=data,
            )
            return {name: value}

        self._registered_feature_names.append(name)
        if dtype is not None:
            self._registered_feature_metadata[name] = (dtype, shape)
        self._registered_feature_funcs.append(feature_func)
        return func

    def get_sampler(self):
        """Return the complete sampler dictionary for Pipeline execution."""
        sampler = dict(self._get_legacy_sampler())
        if not self._parameter_samplers:
            return sampler
        # Append parameter samplers after both active samplers and the
        # config's reserved sampler key range. This prevents optional inactive
        # transitional samplers from having their semantic keys reused accidentally.
        reserved_key_limit = getattr(self, '_sampler_key_limit', -1)
        next_key = max(max(sampler.keys(), default=-1), reserved_key_limit) + 1
        for offset, parameter_sampler in enumerate(self._parameter_samplers):
            sampler[next_key + offset] = parameter_sampler
        return sampler

    def _get_config_hook(self, name):
        """Return a subclass hook method if the config class defines it."""
        if not any(name in cls.__dict__ for cls in type(self).__mro__):
            return None
        return getattr(self, name)

    def get_feature_collection(self, features, f, num):
        """Build the feature collection for this config."""
        features = [] if features is None else list(features)
        feature_instances = [feat for feat in features if isinstance(feat, BaseFeatureCatalog)]
        default_feature_names = [feat for feat in features if isinstance(feat, str)]
        feature_instances += self.get_default_features(default_feature_names, f, num)
        builder = BaseFeatureCollectionBuilder(features=feature_instances, parameters=self.parameters)
        for prepare_func in self._registered_prepare_funcs:
            builder.add_custom(prepare_func)
        prepare_hook = self._get_config_hook('get_prepare_func')
        if prepare_hook is not None:
            builder.add_custom(prepare_hook())
        feature_collection = builder.build()
        for name, (dtype, shape) in self._registered_feature_metadata.items():
            builder.add_mapper(name, dtype, shape)
        for feature_func in self._registered_feature_funcs:
            builder.add_custom(feature_func)
        cleanup_hook = self._get_config_hook('get_cleanup_func')
        if cleanup_hook is not None:
            builder.add_custom(cleanup_hook(features))
        return feature_collection

    def configure_pipeline(self, pipeline, features, f, num):
        """Attach this config's samplers and feature functions to a Pipeline."""
        feature_collection = self.get_feature_collection(features, f, num)
        pipeline.sampler = self.get_sampler()
        pipeline.features = feature_collection.get_feature_funcs()
        return feature_collection

    def _get_default_feature_kwargs(self, f, num):
        """Return keyword arguments passed to default feature builder methods."""
        return {'f': f, 'num': num}

    def get_default_features(self, features, f, num):
        """
        Build default features using `_get_default_feature_{name}` methods.

        Parameters
        ----------
        features : list[str]
            Names of default features to include.
        f : float | list[float] | None
            Frequencies used for frequency-dependent features.
        num : int
            Bandwidth selector for fractional octave features.

        Returns
        -------
        list
            Instantiated feature catalog objects.
        """
        builder_kwargs = self._get_default_feature_kwargs(f, num)
        default_features = []
        registered_feature_names = set(self._registered_feature_names)
        for feature_name in features:
            if feature_name not in ['idx', 'seeds'] and feature_name not in registered_feature_names:
                builder = getattr(self, f'_get_default_feature_{feature_name}', None)
                if builder is None:
                    msg = f'Unknown feature "{feature_name}".'
                    raise ValueError(msg)
                default_features.append(builder(**builder_kwargs))
        return default_features


class Dataset(HasPrivateTraits):
    """
    Class for generating microphone array datasets with specified features and labels.

    Attributes
    ----------
    config : Config
        Configuration object for dataset generation.
    tasks : int
        Number of parallel tasks for data generation. Defaults to 1 (sequential calculation).
    """

    config = Instance(Config, desc='configuration object')
    tasks = Property(desc='number of parallel tasks for data generation')
    remote_args = Dict({})
    #: logger instance to log calculation times for each data sample
    logger = Property(desc='Logger instance to log timing statistics')

    # private
    _logger = Instance(logging.Logger, desc='Internal logger instance')
    _tasks = Int(1, desc='number of parallel tasks for data generation')

    def __init__(self, config=None, tasks=1, remote_args=None, logger=None):
        HasPrivateTraits.__init__(self)
        self.tasks = tasks
        if config is None:
            config = Config()
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
            return DistributedPipeline(numworkers=self.tasks, remote_args=self.remote_args)
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

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        return self.config.get_feature_collection(features, f, num)

    def generate(self, features=None, size=None, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Generate dataset samples iteratively.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
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
        if size is None:
            msg = 'Dataset.generate() requires a size.'
            raise ValueError(msg)
        pipeline = self.get_pipeline_instance()
        self.config.configure_pipeline(pipeline, features, f, num)
        set_pipeline_seeds(pipeline, start_idx, size, split)
        if not pipeline.random_seeds:
            pipeline.numsamples = size
        yield from pipeline.get_data(progress_bar=progress_bar, start_idx=start_idx)

    def save_h5(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a HDF5 file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
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
            dataset = DatasetSynthetic().save_h5(
                f=f, num=num, split='training', size=10, features=features, name='/tmp/example.h5'
            )
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        self.config.configure_pipeline(pipeline, features, f, num)
        set_pipeline_seeds(pipeline, start_idx, size, split)
        if not pipeline.random_seeds:
            pipeline.numsamples = size
        WriteH5Dataset(
            name=name,
            source=pipeline,
        ).save(progress_bar, start_idx)  # start the calculation


ConfigBase = Config
DatasetBase = Dataset


if TF_FLAG:
    import tensorflow as tf

    from acoupipe.writer import WriteTFRecord, complex_list_feature

    def save_tfrecord(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a .tfrecord file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
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
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        feature_collection = self.config.configure_pipeline(pipeline, features, f, num)
        set_pipeline_seeds(pipeline, start_idx, size, split)
        if not pipeline.random_seeds:
            pipeline.numsamples = size
        # get features with varying length to handle them correctly in the TFRecord writer
        shape_features = []
        for feature, shape in feature_collection.feature_tf_shape_mapper.items():
            # if more then one None in shape, we have a varying length feature
            if list(shape).count(None) > 1:
                shape_features.append(feature)

        WriteTFRecord(
            name=name,
            source=pipeline,
            shape_features=shape_features,
            encoder_funcs=feature_collection.feature_tf_encoder_mapper,
        ).save(
            progress_bar,
            start_idx,
        )

    Dataset.save_tfrecord = save_tfrecord

    def get_output_signature(self, features, f=None, num=0):
        """Get the output signature of the dataset.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
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
        signature = {}
        feature_collection = self.get_feature_collection(features, f, num)
        for feature in features:
            signature[feature] = tf.TensorSpec(
                feature_collection.feature_tf_shape_mapper[feature],
                feature_collection.feature_tf_dtype_mapper[feature],
            )
        return signature

    Dataset.get_output_signature = get_output_signature

    def get_tf_dataset(self, features, size, split='training', f=None, num=0, start_idx=0, progress_bar=False):
        """Get a TensorFlow dataset from the generated data.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
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
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        self.config.configure_pipeline(pipeline, features, f, num)
        set_pipeline_seeds(pipeline, start_idx, size, split)
        if not pipeline.random_seeds:
            pipeline.numsamples = size
        features = features + ['idx', 'seeds']
        output_signature = self.get_output_signature(features, f=f, num=num)
        return tf.data.Dataset.from_generator(
            partial(self._generate, pipeline=pipeline, start_idx=start_idx, progress_bar=progress_bar),
            output_signature=output_signature,
        )

    Dataset.get_tf_dataset = get_tf_dataset

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
        feature_collection = self.get_feature_collection(features, f, num)
        features = features + ['idx', 'seeds']

        feature_description = {}
        shapes = dict(feature_collection.feature_tf_shape_mapper)  # make a copy

        for feature in features:
            dtype = feature_collection.feature_tf_dtype_mapper[feature]
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

                encoder = feature_collection.feature_tf_encoder_mapper[feature]
                if encoder is complex_list_feature:
                    value = tf.complex(value[..., 0], value[..., 1])

                data[feature] = value
            return data

        return _parse_function

    Dataset.get_tfrecord_parser = get_tfrecord_parser
