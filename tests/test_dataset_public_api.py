# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import numpy as np
from scipy.stats import norm

from acoupipe.datasets import Config as PublicConfig
from acoupipe.datasets import Dataset as PublicDataset
from acoupipe.datasets import ParameterSet as PublicParameterSet
from acoupipe.datasets.base import Config, ConfigBase, Dataset, DatasetBase
from acoupipe.datasets.features import create_feature
from acoupipe.datasets.parameters import ParameterSet
from acoupipe.sampler import AttributeSampler


def rms_parameters():
    return ParameterSet.from_dict({'rms': 1.0})


def test_new_dataset_and_config_names_keep_base_aliases():
    assert ConfigBase is Config
    assert DatasetBase is Dataset
    assert PublicConfig is Config
    assert PublicDataset is Dataset
    assert PublicParameterSet is ParameterSet

    config = Config(parameters=rms_parameters())
    dataset = Dataset(config=config)

    assert isinstance(config, Config)
    assert isinstance(dataset, Dataset)


def test_config_defaults_to_empty_parameter_set():
    config = Config()

    assert isinstance(config.parameters, ParameterSet)
    assert config.parameters.to_dict() == {}


def test_config_sample_supports_direct_parameter_set_attributes():
    parameters = rms_parameters()
    config = Config(parameters=parameters)

    sampler = config.sample('rms', random_var=norm(loc=2.0, scale=0.0))

    assert isinstance(sampler, AttributeSampler)
    assert sampler.target is parameters
    assert sampler.attribute == 'rms'

    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert parameters.rms == 2.0
    assert sampler.value == 2.0


def test_config_sample_accepts_numpy_random_function():
    parameters = rms_parameters()
    config = Config(parameters=parameters)

    sampler = config.sample('rms', random_func=lambda rng: rng.rayleigh(scale=5.0))
    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert isinstance(sampler, AttributeSampler)
    assert parameters.rms == np.random.default_rng(1).rayleigh(scale=5.0)


def test_config_sample_passes_parameters_to_random_function_for_ordered_dependencies():
    parameters = ParameterSet.from_dict({'nsources': 1, 'rms': np.array([])})
    config = Config(parameters=parameters)

    config.sample('nsources', random_func=lambda _, params: params.nsources + 2)
    config.sample('rms', random_func=lambda _, params: np.ones(params.nsources))

    sample = next(Dataset(config=config).generate(size=1, progress_bar=False))

    assert sample['idx'] == 0
    np.testing.assert_equal(parameters.rms, np.ones(3))


def test_config_sample_rejects_unknown_parameter_path():
    config = Config(parameters=rms_parameters())

    with np.testing.assert_raises_regex(ValueError, 'Unsupported sampling path "unknown"'):
        config.sample('unknown', random_var=norm(loc=2.0, scale=0.0))


def test_dataset_and_config_can_generate_from_parameter_sampling_without_subclassing():
    parameters = rms_parameters()
    config = Config(parameters=parameters)
    config.sample('rms', random_var=norm(loc=2.0, scale=0.0))

    feature = create_feature(
        feature_func=lambda _: {'rms': parameters.rms},
        name='rms',
        shape=(),
        dtype=np.float32,
    )

    sample = next(Dataset(config=config).generate(features=[feature], size=1, progress_bar=False))

    assert sample['rms'] == 2.0


def test_config_convenience_api_generates_without_explicit_feature_list():
    parameters = rms_parameters()
    runtime = {}
    config = Config(parameters=parameters)

    config.sample('rms', random_var=norm(loc=2.0, scale=0.0))
    config.prepare(lambda parameters: runtime.update(rms=parameters.rms))
    config.feature('rms', lambda: runtime['rms'])
    config.feature('rms_from_parameters', lambda parameters: parameters.rms)
    config.feature('rms_squared', lambda data: data['rms'] ** 2)

    sample = next(Dataset(config=config).generate(size=1, progress_bar=False))

    assert sample['rms'] == 2.0
    assert sample['rms_from_parameters'] == 2.0
    assert sample['rms_squared'] == 4.0


def test_config_prepare_runs_before_legacy_prepare_and_features():
    order = []

    class LegacyPrepareConfig(Config):
        def get_prepare_func(self):
            def legacy_prepare(sampler):  # noqa: ARG001
                order.append('legacy')
                return {}

            return legacy_prepare

    config = LegacyPrepareConfig(parameters=rms_parameters())
    config.prepare(lambda: order.append('registered'))
    config.feature('order', lambda: order)

    sample = next(Dataset(config=config).generate(size=1, progress_bar=False))

    assert sample['order'] == ['registered', 'legacy']


def test_config_feature_metadata_supports_explicit_feature_list_and_tf_dataset():
    import tensorflow as tf

    config = Config(parameters=rms_parameters())
    config.feature('rms', lambda parameters: np.float32(parameters.rms), dtype=np.float32, shape=())
    dataset = Dataset(config=config)

    sample = next(dataset.generate(features=['rms'], size=1, progress_bar=False))
    tf_sample = next(iter(dataset.get_tf_dataset(features=['rms'], size=1, progress_bar=False)))

    assert sample['rms'] == np.float32(1.0)
    assert tf_sample['rms'].dtype == tf.float32
    assert tf_sample['rms'].shape == ()


def test_config_feature_metadata_supports_h5_saving(tmp_path):
    import h5py

    config = Config(parameters=rms_parameters())
    config.feature('rms', lambda parameters: np.float32(parameters.rms), dtype=np.float32, shape=())
    output = tmp_path / 'custom.h5'

    Dataset(config=config).save_h5(features=['rms'], size=1, name=output, progress_bar=False)

    with h5py.File(output, 'r') as h5:
        assert h5['0/rms'][()] == np.float32(1.0)


def test_config_feature_metadata_supports_tfrecord_saving_and_parsing(tmp_path):
    import tensorflow as tf

    config = Config(parameters=rms_parameters())
    config.feature('rms', lambda parameters: np.float32(parameters.rms), dtype=np.float32, shape=())
    dataset = Dataset(config=config)
    output = tmp_path / 'custom.tfrecord'

    dataset.save_tfrecord(features=['rms'], size=1, name=output, progress_bar=False)
    parsed = tf.data.TFRecordDataset(output.as_posix()).map(
        dataset.get_tfrecord_parser(features=['rms'], f=None, num=0)
    )
    sample = next(iter(parsed))

    assert sample['rms'].numpy() == np.float32(1.0)
