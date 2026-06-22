"""Unit tests for the high-level Dataset Config API."""

# ruff: noqa: S101

from acoupipe.datasets import Config, ConfigBase, Dataset, DatasetBase, ParameterSet
from acoupipe.sampler import AttributeSampler

import numpy as np
from scipy.stats import norm


def test_public_config_and_dataset_keep_legacy_base_classes():
    """Test that new public names do not replace legacy base classes."""
    assert Config is not ConfigBase
    assert Dataset is DatasetBase


def test_config_sample_registers_parameter_sampler():
    """Test that Config.sample mutates a direct ParameterSet attribute."""
    parameters = ParameterSet.from_dict({'snr_db': 10.0})
    config = Config(parameters=parameters)

    sampler = config.sample('snr_db', random_var=norm(loc=20.0, scale=0.0))
    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert isinstance(sampler, AttributeSampler)
    assert parameters.snr_db == 20.0


def test_parameter_attributes_are_default_features_without_registration():
    """Test that direct ParameterSet attributes can be requested as features."""
    parameters = ParameterSet.from_dict({'c': 343.0, 'snr_db': 20.0})
    config = Config(parameters=parameters)

    sample = next(Dataset(config=config).generate(features=['c', 'snr_db'], size=1, progress_bar=False))

    assert sample['c'] == 343.0
    assert sample['snr_db'] == 20.0


def test_sampled_parameter_feature_reflects_sampled_value():
    """Test that ParameterSet-backed features see values after sampling."""
    parameters = ParameterSet.from_dict({'snr_db': 10.0})
    config = Config(parameters=parameters)
    config.sample('snr_db', random_var=norm(loc=20.0, scale=0.0))

    sample = next(Dataset(config=config).generate(features=['snr_db'], size=1, progress_bar=False))

    assert sample['snr_db'] == 20.0


def test_registered_feature_can_depend_on_requested_parameter_feature():
    """Test that registered features can consume earlier requested data."""
    parameters = ParameterSet.from_dict({'snr_db': 20.0})
    config = Config(parameters=parameters)
    config.feature('snr_linear', lambda data: 10 ** (data['snr_db'] / 10))

    sample = next(Dataset(config=config).generate(features=['snr_db', 'snr_linear'], size=1, progress_bar=False))

    assert sample['snr_db'] == 20.0
    assert sample['snr_linear'] == 100.0
