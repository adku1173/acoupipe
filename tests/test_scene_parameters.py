# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import numpy as np
import pytest
from scipy.stats import norm

from acoupipe.datasets.base import Dataset
from acoupipe.datasets.parameters import MIRACLEParameters, ParameterSet, SyntheticParameters
from acoupipe.datasets.synthetic import DatasetSynthetic
from acoupipe.sampler import AttributeSampler


def test_scene_parameters_extend_parameter_set():
    parameters = SyntheticParameters()

    assert isinstance(parameters, ParameterSet)
    assert parameters.c == 343.0
    assert not hasattr(parameters, 'env')


def test_config_sampling_c_creates_attribute_sampler():
    dataset = DatasetSynthetic()
    sampler = dataset.config.sample('c', random_var=norm(loc=340.0, scale=0.0))

    assert isinstance(sampler, AttributeSampler)
    assert sampler.target is dataset.config.parameters
    assert sampler.attribute == 'c'
    assert sampler.equal_value is True

    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert dataset.config.parameters.c == 340.0


def test_config_sampling_rejects_unsupported_path():
    dataset = DatasetSynthetic()

    with pytest.raises(ValueError, match='Unsupported sampling path "mics"'):
        dataset.config.sample('mics', random_var=norm(loc=0.0, scale=1.0))


def test_scene_parameters_include_sourcemap_namespace():
    parameters = SyntheticParameters()

    assert isinstance(parameters.sourcemap, ParameterSet)
    assert parameters.sourcemap.c == 343.0


def test_config_sampling_supports_nested_parameter_path():
    dataset = DatasetSynthetic()
    sampler = dataset.config.sample('sourcemap.c', random_var=norm(loc=350.0, scale=0.0))

    assert isinstance(sampler, AttributeSampler)
    assert sampler.target is dataset.config.parameters
    assert sampler.attribute == 'sourcemap.c'

    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert dataset.config.parameters.sourcemap.c == 350.0


def test_config_sampling_rejects_unknown_nested_parameter_path():
    dataset = DatasetSynthetic()

    with pytest.raises(ValueError, match='Unsupported sampling path "sourcemap.missing"'):
        dataset.config.sample('sourcemap.missing', random_var=norm(loc=0.0, scale=1.0))


def test_dataset_synthetic_exposes_parameters_object():
    dataset = DatasetSynthetic()

    assert dataset.config.parameters.c == 343.0
    assert not hasattr(dataset.config, 'scene')


def test_config_privately_merges_parameter_sampling_into_pipeline_sampler():
    dataset = DatasetSynthetic()
    dataset.config.sample('c', random_var=norm(loc=340.0, scale=0.0))

    legacy_sampler = dataset.config._get_legacy_sampler()
    sampler = dataset.config._get_sampler()

    assert max(sampler) > 6
    assert max(sampler) > max(legacy_sampler)
    assert any(
        isinstance(sampler_, AttributeSampler)
        and sampler_.target is dataset.config.parameters
        and sampler_.attribute == 'c'
        for sampler_ in sampler.values()
    )


def test_sampler_access_is_private_to_config_not_dataset():
    dataset = DatasetSynthetic()

    assert callable(dataset.config._get_sampler)
    assert not hasattr(dataset.config, 'get_sampler')
    assert '_get_sampler' not in Dataset.__dict__


def test_sampler_key_limit_is_private():
    dataset = DatasetSynthetic()

    assert hasattr(dataset.config, '_sampler_key_limit')
    assert not hasattr(dataset.config, 'sampler_key_limit')


def test_parameter_sampling_is_appended_after_reserved_sampler_keys_when_optional_samplers_are_disabled():
    dataset = DatasetSynthetic(mic_sig_noise=False, random_signal_length=False)
    parameter_sampler = dataset.config.sample('c', random_var=norm(loc=340.0, scale=0.0))

    sampler = dataset.config._get_sampler()

    assert 5 not in sampler
    assert 6 not in sampler
    assert sampler[7] is parameter_sampler


def test_dataset_synthetic_uses_sampled_parameter_speed_of_sound_in_runtime_env():
    dataset = DatasetSynthetic()
    dataset.config.sample('c', random_var=norm(loc=340.0, scale=0.0))

    next(dataset.generate(features=[], size=1, progress_bar=False))

    assert dataset.config.parameters.c == 340.0
    assert dataset.config.env.c == 340.0


def test_dataset_synthetic_default_sourcemap_uses_independent_analysis_beamformer():
    dataset = DatasetSynthetic(mic_sig_noise=False)

    feature = dataset.config.get_default_features(['sourcemap'], f=1000.0, num=0)[0]

    assert feature.beamformer.freq_data is dataset.config.freq_data
    assert feature.beamformer is not dataset.config.beamformer
    assert feature.beamformer.steer.env is not dataset.config.env
    assert feature.beamformer.steer.env.c == dataset.config.parameters.sourcemap.c


def test_dataset_synthetic_sourcemap_prepare_uses_sourcemap_speed_not_simulation_speed():
    dataset = DatasetSynthetic(mic_sig_noise=False)
    dataset.config.parameters.c = 340.0
    dataset.config.parameters.sourcemap.c = 350.0
    feature = dataset.config.get_default_features(['sourcemap'], f=1000.0, num=0)[0]

    prepare_func = feature.get_prepare_func()
    prepare_func(parameters=dataset.config.parameters)

    assert dataset.config.parameters.c == 340.0
    assert feature.beamformer.steer.env.c == 350.0


def test_dataset_miracle_uses_measured_c0_as_analysis_default_without_simulation_env(create_miracle_dataset):
    dataset = create_miracle_dataset(full=False, mic_sig_noise=False)

    feature = dataset.config.get_default_features(['sourcemap'], f=1000.0, num=0)[0]

    assert isinstance(dataset.config.parameters, MIRACLEParameters)
    assert dataset.config.measured_c0 == 344.8
    assert not hasattr(dataset.config.parameters, 'c')
    assert dataset.config.env is None
    assert dataset.config.parameters.sourcemap.c == dataset.config.measured_c0
    assert dataset.config.steer.env.c == dataset.config.measured_c0
    assert feature.beamformer.steer.env.c == dataset.config.measured_c0


def test_dataset_miracle_rejects_simulation_speed_sampling(create_miracle_dataset):
    dataset = create_miracle_dataset(full=False, mic_sig_noise=False)

    with pytest.raises(ValueError, match='Unsupported sampling path "c"'):
        dataset.config.sample('c', random_var=norm(loc=340.0, scale=0.0))


def test_dataset_miracle_allows_sourcemap_speed_sampling(create_miracle_dataset):
    dataset = create_miracle_dataset(full=False, mic_sig_noise=False)
    sampler = dataset.config.sample('sourcemap.c', random_var=norm(loc=350.0, scale=0.0))

    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert dataset.config.parameters.sourcemap.c == 350.0


def test_dataset_miracle_preserves_user_supplied_sourcemap_speed():
    from acoupipe.datasets.parameters import MIRACLEParameters
    from tests.miracle_test_config import DatasetMIRACLETestConfig

    parameters = MIRACLEParameters(sourcemap=ParameterSet(c=350.0))
    config = DatasetMIRACLETestConfig(parameters=parameters, mic_sig_noise=False)

    assert config.measured_c0 == 344.8
    assert config.parameters.sourcemap.c == 350.0
    assert config.steer.env.c == 350.0


def test_dataset_synthetic_ism_prepare_functions_receive_parameters(monkeypatch):
    from acoupipe.datasets.synthetic import DatasetSyntheticISMConfig

    monkeypatch.setattr(DatasetSyntheticISMConfig, 'create_acoular_pipeline', lambda self: None)
    config = DatasetSyntheticISMConfig(mode='welch')
    prepare_func = config.get_prepare_func()

    assert prepare_func.keywords['parameters'] is config.parameters

    monkeypatch.undo()
    config = DatasetSyntheticISMConfig(mode='analytic')
    prepare_func = config.get_prepare_func()

    assert prepare_func.keywords['parameters'] is config.parameters


def test_dataset_synthetic_ism_prepare_ir_forwards_speed_of_sound(monkeypatch):
    from acoupipe.datasets import synthetic
    from acoupipe.datasets.synthetic import DatasetSyntheticISMConfig

    captured = {}

    def fake_get_ir(sample_freq, room_dim, mloc, sloc, rt60, c=343.0):  # noqa: ARG001
        captured['c'] = c
        return [np.ones((1, 3)), np.full((1, 3), 2.0)]

    def fake_calc_transfer(ir, sample_freq, block_size, fftfreq):  # noqa: ARG001
        return np.ones(fftfreq.shape[0], dtype=complex)

    class FreqData:
        sample_freq = 51200
        block_size = 128

        def fftfreq(self):
            return np.array([0.0, 1000.0])

    class Mics:
        num_mics = 1
        pos = np.array([[0.0], [0.0], [0.0]])
        pos_total = pos

    monkeypatch.setattr(synthetic, 'get_ir', fake_get_ir)
    monkeypatch.setattr(synthetic, 'calc_transfer', fake_calc_transfer)

    DatasetSyntheticISMConfig._prepare_ir(
        mics=Mics(),
        freq_data=FreqData(),
        loc=np.array([[1.0], [0.0], [0.0]]),
        ref_loc=np.array([0.0, 0.0, 0.0]),
        room_params={'room_size': [6, 4, 3], 'rt60': 2.0},
        c=340.0,
        domain='frequency',
    )

    assert captured['c'] == 340.0
