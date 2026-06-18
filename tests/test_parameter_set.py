# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import json

import numpy as np
import pytest
from scipy.stats import norm

from acoupipe.datasets import Config, Dataset, ParameterSet
from acoupipe.sampler import AttributeSampler


def test_parameter_set_from_dict_exposes_name_value_mapping_as_attributes():
    parameters = ParameterSet.from_dict({'rms': 1.0, 'block_size': 128})

    assert parameters.rms == 1.0
    assert parameters.block_size == 128
    assert parameters.to_dict() == {'rms': 1.0, 'block_size': 128}


def test_parameter_set_allows_plain_assignment_without_schema_casting():
    parameters = ParameterSet.from_dict({'rms': 1.0})

    parameters.rms = 2.5

    assert parameters.rms == 2.5


def test_parameter_set_loads_name_value_mappings_from_json_and_toml_files(tmp_path):
    json_path = tmp_path / 'parameters.json'
    json_path.write_text(json.dumps({'rms': 1.0}), encoding='utf-8')
    toml_path = tmp_path / 'parameters.toml'
    toml_path.write_text('rms = 2.0\n', encoding='utf-8')

    assert ParameterSet.from_json(json_path).rms == 1.0
    assert ParameterSet.from_toml(toml_path).rms == 2.0


def test_parameter_set_rejects_invalid_parameter_names():
    with pytest.raises(ValueError, match='Invalid parameter name'):
        ParameterSet.from_dict({'not-a-python-attribute': 1.0})


def test_config_sample_registers_attribute_sampler_for_parameter_set_attribute():
    parameters = ParameterSet.from_dict({'rms': 1.0})
    config = Config(parameters=parameters)

    sampler = config.sample('rms', random_var=norm(loc=2.0, scale=0.0))

    assert isinstance(sampler, AttributeSampler)
    assert sampler.target is parameters
    assert sampler.attribute == 'rms'

    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert parameters.rms == 2.0
    assert sampler.value == 2.0


def test_config_convenience_api_generates_with_parameter_set():
    parameters = ParameterSet.from_dict({'rms': 1.0})
    config = Config(parameters=parameters)

    config.sample('rms', random_var=norm(loc=2.0, scale=0.0))
    config.feature('rms', lambda parameters: parameters.rms)
    config.feature('rms_squared', lambda data: data['rms'] ** 2)

    sample = next(Dataset(config=config).generate(size=1, progress_bar=False))

    assert sample['rms'] == 2.0
    assert sample['rms_squared'] == 4.0
