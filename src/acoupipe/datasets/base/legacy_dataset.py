"""Legacy dataset base class.

:class:`DatasetBase` keeps the legacy generation API with ``f`` and ``num`` as method
arguments. It is used with :class:`~acoupipe.datasets.base.legacy_config.ConfigBase`
and will be removed together with the legacy configurations.
"""

from functools import partial

from acoupipe.datasets._shared.utils import set_pipeline_seed
from acoupipe.datasets.base.dataset import Dataset
from acoupipe.datasets.base.legacy_config import ConfigBase
from acoupipe.datasets.features import get_tf_mappers
from acoupipe.writer import WriteH5Dataset

from traits.api import Any


class DatasetBase(Dataset):
    """
    Legacy base class for generating microphone array datasets with specified features and labels.

    :class:`DatasetBase` is used with configurations based on
    :class:`~acoupipe.datasets.base.legacy_config.ConfigBase`, which receive ``f`` and ``num``
    as method arguments. For a :class:`~acoupipe.datasets.base.config.Config`, use
    :class:`~acoupipe.datasets.base.dataset.Dataset` and set ``config.parameters.f`` and
    ``config.parameters.num`` instead.

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

    def __init__(self, config=None, tasks=1, remote_args=None, logger=None, samples_per_task=16, seed=0):
        if config is None:
            config = ConfigBase()
        elif not (hasattr(config, 'get_sampler') and hasattr(config, 'get_features')):
            msg = 'config must provide get_sampler() and get_features()'
            raise TypeError(msg)
        super().__init__(
            config=config,
            tasks=tasks,
            remote_args=remote_args,
            logger=logger,
            samples_per_task=samples_per_task,
            seed=seed,
        )

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
        pipeline.feature_func, _ = self.config.get_features(features, f, num)
        set_pipeline_seed(pipeline, size, split, self.seed)
        yield from pipeline.get_data(progress_bar=progress_bar, start_idx=start_idx)

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
        pipeline.feature_func, _ = self.config.get_features(features, f, num)
        set_pipeline_seed(pipeline, size, split, self.seed)
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
        pipeline.feature_func, feature_objects = self.config.get_features(features, f, num)
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
        _, feature_objects = self.config.get_features(features, f, num)
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
        pipeline.feature_func, _ = self.config.get_features(features, f, num)
        set_pipeline_seed(pipeline, size, split, self.seed)
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

        _, feature_objects = self.config.get_features(features, f, num)
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


__all__ = ['DatasetBase']
