"""Base classes for generating microphone array datasets."""

import logging
from functools import partial

from traits.api import Float, HasPrivateTraits, Instance, Int, Property

from acoupipe.config import TF_FLAG
from acoupipe.datasets.collection import BaseFeatureCollection, FeatureCollectionBuilder
from acoupipe.datasets.features import Feature
from acoupipe.datasets.setup import MsmSetupBase, SamplerSetupBase
from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.writer import WriteH5Dataset

if TF_FLAG:
    import tensorflow as tf

    from acoupipe.writer import WriteTFRecord, complex_list_feature

class ConfigBase(HasPrivateTraits):
    """Configuration base class for generating AcouPipe datasets."""

    msm_setup = Instance(MsmSetupBase, desc="simulation setup object")
    sampler_setup = Instance(SamplerSetupBase, args=(), desc="sampler setup object")
    _feature_builder = Instance(FeatureCollectionBuilder, args=(), desc="feature setup object")

    def get_feature_collection(self, features, f=None, num=0):
        """
        Get the feature collection of the dataset.

        Parameters
        ----------
        features : list
            List of features included in the dataset.
        f : float
            The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to 0 (single frequency line).

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        self._feature_builder.feature_collection = BaseFeatureCollection()
        self._feature_builder.add_idx()
        self._feature_builder.add_seeds(self.sampler_setup.numsampler)
        for feature in features:
            if isinstance(feature, Feature):
                self._feature_builder.add_feature(feature)
            else:
                raise ValueError(f"Feature {feature} is not a valid feature object.")
        return self._feature_builder.feature_collection

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.sampler.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The key defines the idx in the sample order.

        e.g.:
        >>> sampler = {
        >>>     0 : BaseSampler(...),
        >>>     1 : BaseSampler(...),
        >>>     ...
        >>> }

        Returns
        -------
        dict
            dictionary containing the sampler objects
        """
        return self.sampler_setup.get_sampler()


class DatasetBase(HasPrivateTraits):
    """
    Base class for generating microphone array datasets with specified features and labels.

    Attributes
    ----------
    config : ConfigBase
        Configuration object for dataset generation.
    tasks : int
        Number of parallel tasks for data generation. Defaults to 1 (sequential calculation).
    """

    config = Instance(ConfigBase, desc="configuration object")
    tasks = Property(desc="number of parallel tasks for data generation")
    num_gpus = Float(0, desc="number of GPUs to use for data generation (only relevant if multiple tasks are used)")

    #: logger instance to log calculation times for each data sample
    logger = Property(desc="Logger instance to log timing statistics")

    #private
    _logger = Instance(logging.Logger, desc="Internal logger instance")
    _tasks = Int(1, desc="number of parallel tasks for data generation")

    def __init__(self,config=None, tasks=1, num_gpus=0, logger=None):
        HasPrivateTraits.__init__(self)
        self.tasks = tasks
        self.num_gpus = num_gpus
        # If no config is provided, use the default ConfigBase settings
        if config is None:
            config = ConfigBase()
        self.config = config
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
        logger.propagate = False # don't propagate to the root logger!
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d %(message)s",
                datefmt="%H:%M:%S"))
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
                numworkers=self.tasks, num_gpus=self.num_gpus)
        else:
            return BasePipeline()

    @staticmethod
    def _generate(pipeline, progress_bar, start_idx):
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
        for data in pipeline.get_data(
            progress_bar=progress_bar, start_idx=start_idx):
            yield data


    def generate(self, features, split, size, f=None, num=0, start_idx=0, progress_bar=True):
        """Generate dataset samples iteratively.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test').
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
        Generate features iteratively.

        >>> from acoupipe.datasets.synthetic import DatasetSynthetic
        >>> # define the features
        >>> features = ["csm", "source_strength_analytic", "loc"]
        >>> f = 1000
        >>> num = 3
        >>> # generate the dataset
        >>> generator = DatasetSynthetic().generate(
                f=f, num=num, split="training", size=2, features=features)
        >>> # iterate over the dataset
        >>> for data in generator:
                print(data)
        """
        pipeline = self.get_pipeline_instance()
        pipeline.sampler = self.config.get_sampler()
        pipeline.features = self.config.get_feature_collection(features, f, num).get_feature_funcs()
        self.config.sampler_setup.set_seeds(pipeline, start_idx, size, split)
        for data in pipeline.get_data(
            progress_bar=progress_bar, start_idx=start_idx):
            yield data

    def save_h5(self, features, split, size, name, f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a HDF5 file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test').
        size : int
            Size of the dataset (number of source cases).
        name : str
            Name of the HDF5 file.
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
        Save features to a HDF5 file.

        >>> from acoupipe.datasets.synthetic import DatasetSynthetic
        >>> # define the features
        >>> features = ["csm", "source_strength_analytic", "loc"]
        >>> f = 1000
        >>> num = 3
        >>> # save the dataset
        >>> dataset = DatasetSynthetic().save_h5(
                f=f, num=num, split="training", size=10, features=features,name="/tmp/example.h5")
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        pipeline.features = self.config.get_feature_collection(features, f, num).get_feature_funcs()
        self.config.sampler_setup.set_seeds(pipeline, start_idx, size, split)
        WriteH5Dataset(name=name,
                       source=pipeline,
                       ).save(progress_bar, start_idx)  # start the calculation


if TF_FLAG:
    import tensorflow as tf

    from acoupipe.writer import WriteTFRecord, complex_list_feature


    def save_tfrecord(self, features, split, size, name, f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a .tfrecord file.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test').
        size : int
            Size of the dataset (number of source cases).
        name : str
            Name of the TFRecord file.
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
        Save features to a TFRecord file.

        >>> from acoupipe.datasets.synthetic import DatasetSynthetic
        >>> # define the features
        >>> features = ["csm", "source_strength_analytic", "loc"]
        >>> f = 1000
        >>> num = 3
        >>> # save the dataset
        >>> dataset = DatasetSynthetic().save_tfrecord(
                f=f, num=num, split="training", size=10, features=features,name="/tmp/example.tfrecord")
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        feature_collection = self.config.get_feature_collection(features, f, num)
        pipeline.features = feature_collection.get_feature_funcs()
        self.config.sampler_setup.set_seeds(pipeline, start_idx, size, split)
        WriteTFRecord(name=name, source=pipeline,
                      encoder_funcs=feature_collection.feature_tf_encoder_mapper).save(progress_bar, start_idx)
    DatasetBase.save_tfrecord = save_tfrecord


    def get_tf_dataset(self, features, split, size, f=None, num=0, start_idx=0, progress_bar=False):
        """Get a TensorFlow dataset from the generated data.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.
        split : str
            Split name for the dataset ('training', 'validation' or 'test').
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
            Whether to show a progress bar (default is False).

        Returns
        -------
        tf.data.Dataset
            TensorFlow dataset containing the generated data. The dataset elements have the structure defined
            by the output_signature, which is based on the shapes of dataset features.
        """
        pipeline = self.get_pipeline_instance()
        # self._setup_logging(pipeline=pipeline)
        pipeline.sampler = self.config.get_sampler()
        feature_collection = self.config.get_feature_collection(features, f, num)
        pipeline.features = feature_collection.get_feature_funcs()
        self.config.sampler_setup.set_seeds(pipeline, start_idx, size, split)
        output_signature = feature_collection.get_output_signature(features + ["idx", "seeds"])
        return tf.data.Dataset.from_generator(
            partial(
                DatasetBase._generate, pipeline=pipeline, start_idx=start_idx, progress_bar=progress_bar),
                output_signature=output_signature)
    DatasetBase.get_tf_dataset = get_tf_dataset

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
        >>> from acoupipe.datasets.synthetic import DatasetSynthetic
        >>> # define the features
        >>> features = ["csm", "source_strength_analytic", "loc"]
        >>> f = 1000
        >>> num = 3
        >>> # save the dataset
        >>> dataset = DatasetSynthetic().save_tfrecord(
                f=f, num=num, split="training", size=10, features=features,name="/tmp/example.tfrecord")
        >>> # parse the dataset
        >>> parser = dataset.get_tfrecord_parser(features, f, num)
        >>> dataset = tf.data.TFRecordDataset("/tmp/example.tfrecord")
        >>> dataset = iter(dataset.map(parser))
        >>> data = next(dataset)

        """
        feature_collection = self.config.get_feature_collection(features, f, num)
        features = features + ["idx", "seeds"]
        feature_description = {}
        shapes = feature_collection.feature_tf_shape_mapper
        for feature in features:
            if feature_collection.feature_tf_encoder_mapper[feature] == complex_list_feature:
                dtype = tf.float32 # complex not supported for tfrecord files
                shapes[feature] = shapes[feature] + (2,)
            else:
                dtype = feature_collection.feature_tf_dtype_mapper[feature]
            if None in shapes[feature]:
                feature_description[feature] = tf.io.VarLenFeature(dtype)
            else:
                feature_description[feature] = tf.io.FixedLenFeature(shapes[feature], dtype)
        # create parser func
        def _parse_function(example_proto):
            data = tf.io.parse_single_example(example_proto, feature_description)
            for feature in features:
                shape = shapes[feature]
                if None in shape:
                    shape = [s if s is not None else -1 for s in shape]
                    data[feature] = tf.reshape(tf.sparse.to_dense(data[feature]),
                                                shape)
                if feature_collection.feature_tf_encoder_mapper[feature] == complex_list_feature: # recover complex dtype
                    data[feature] = tf.complex(data[feature][...,0],data[feature][...,1])
            return data
        return _parse_function
    DatasetBase.get_tfrecord_parser = get_tfrecord_parser

