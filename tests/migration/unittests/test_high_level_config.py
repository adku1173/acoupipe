"""Unit tests for the high-level Dataset Config API.

The tests in this module guard the migration-only API surface that sits next
to the legacy dataset configuration classes. They use ``pytest-cases`` for
case-driven coverage so each dataset family and sampled parameter scenario has
an explicit, documented case function.
"""

# ruff: noqa: S101

from acoupipe.datasets import (
    Config,
    ConfigBase,
    Dataset,
    DatasetBase,
    MIRACLEConfig,
    ParameterSet,
    SRIRACHAConfig,
    SyntheticConfig,
    SyntheticISMConfig,
)
from acoupipe.sampler import AttributeSampler

import numpy as np
import pytest
from pytest_cases import case, parametrize_with_cases
from scipy.stats import norm


@case(id='synthetic')
def case_zero_processing_synthetic_feature_names():
    """Return shallow Synthetic parameters exposed as direct features.

    This case lists only first-pass shallow simulation state. It intentionally
    excludes legacy output aliases such as ``loc`` and any processed Acoular
    features.
    """
    return SyntheticConfig, ('c', 'mics', 'fs', 'signal_length', 'source_locations', 'source_rms', 'snr_db')


@case(id='synthetic-ism')
def case_zero_processing_synthetic_ism_feature_names():
    """Return shallow Synthetic ISM parameters exposed as direct features.

    Synthetic ISM inherits the synthetic simulation fields and adds room
    acoustics state. The case proves that inheritance works for zero-processing
    ParameterSet-backed features.
    """
    return SyntheticISMConfig, (
        'c',
        'mics',
        'fs',
        'signal_length',
        'source_locations',
        'source_rms',
        'snr_db',
        'rt60',
        'room_size',
    )


@case(id='miracle')
def case_zero_processing_miracle_feature_names():
    """Return shallow MIRACLE parameters exposed as direct features.

    The measured-dataset schema contains provenance and sample metadata only.
    Scenario-derived runtime state, such as measured microphone geometry, is
    intentionally absent from this case.
    """
    return MIRACLEConfig, ('scenario', 'fs', 'signal_length', 'source_locations', 'source_rms', 'snr_db')


@case(id='sriracha')
def case_zero_processing_sriracha_feature_names():
    """Return shallow SRIRACHA parameters exposed as direct features.

    SRIRACHA shares the measured first-pass schema with MIRACLE while using its
    own scenario defaults and vocabulary.
    """
    return SRIRACHAConfig, ('scenario', 'fs', 'signal_length', 'source_locations', 'source_rms', 'snr_db')


@case(id='synthetic-snr-db')
def case_sampled_shallow_synthetic_snr_db():
    """Return a sampled synthetic SNR parameter case.

    The sampled value should be visible when ``snr_db`` is requested as a direct
    zero-processing feature.
    """
    return SyntheticConfig, 'snr_db', 12.0


@case(id='synthetic-ism-rt60')
def case_sampled_shallow_synthetic_ism_rt60():
    """Return a sampled Synthetic ISM reverberation-time case.

    The case verifies that dataset-specific shallow parameters added by a
    subclass are sampled before the feature value is copied to the output.
    """
    return SyntheticISMConfig, 'rt60', 1.25


@case(id='miracle-signal-length')
def case_sampled_shallow_miracle_signal_length():
    """Return a sampled MIRACLE signal-length case.

    ``signal_length`` is sample metadata in the measured first-pass schema and
    should behave like any other direct ParameterSet-backed feature.
    """
    return MIRACLEConfig, 'signal_length', 2.5


@case(id='sriracha-scenario')
def case_sampled_shallow_sriracha_scenario():
    """Return a sampled SRIRACHA scenario case.

    Scenario sampling uses a custom random function to avoid numeric-only
    distribution assumptions while still exercising the sampler pipeline.
    """
    return SRIRACHAConfig, 'scenario', 'SR2-D'


def test_public_config_and_dataset_keep_legacy_base_classes():
    """Test that new public names do not replace legacy base classes.

    The high-level ``Config`` and ``Dataset`` names are additive migration API.
    Legacy ``ConfigBase`` and ``DatasetBase`` must remain distinct so existing
    subclass hooks continue to target the old implementation.
    """
    assert Config is not ConfigBase
    assert Dataset is DatasetBase


def test_config_sample_registers_parameter_sampler():
    """Test that ``Config.sample`` mutates a direct ParameterSet attribute.

    The returned sampler should be an ``AttributeSampler`` targeting the
    supplied ``ParameterSet``. Calling the sampler directly is enough to prove
    that the configured path updates ``parameters.snr_db``.
    """
    parameters = ParameterSet.from_dict({'snr_db': 10.0})
    config = Config(parameters=parameters)

    sampler = config.sample('snr_db', random_var=norm(loc=20.0, scale=0.0))
    sampler.random_state = np.random.default_rng(1)
    sampler.sample()

    assert isinstance(sampler, AttributeSampler)
    assert parameters.snr_db == 20.0


def test_parameter_attributes_are_default_features_without_registration():
    """Test direct ParameterSet attributes as default features.

    New high-level configs expose shallow parameter values without requiring
    users to register a matching ``config.feature`` callback. This test covers
    the generic ``ParameterSet`` path used before dataset-specific configs are
    involved.
    """
    parameters = ParameterSet.from_dict({'c': 343.0, 'snr_db': 20.0})
    config = Config(parameters=parameters)

    sample = next(Dataset(config=config).generate(features=['c', 'snr_db'], size=1, progress_bar=False))

    assert sample['c'] == 343.0
    assert sample['snr_db'] == 20.0


def test_sampled_parameter_feature_reflects_sampled_value():
    """Test that ParameterSet-backed features see values after sampling.

    Generation runs samplers before feature functions. A requested parameter
    feature must therefore copy the sampled ``snr_db`` value, not the initial
    value stored on the ``ParameterSet``.
    """
    parameters = ParameterSet.from_dict({'snr_db': 10.0})
    config = Config(parameters=parameters)
    config.sample('snr_db', random_var=norm(loc=20.0, scale=0.0))

    sample = next(Dataset(config=config).generate(features=['snr_db'], size=1, progress_bar=False))

    assert sample['snr_db'] == 20.0


def test_registered_feature_can_depend_on_requested_parameter_feature():
    """Test registered features can consume prior parameter feature data.

    The requested ``snr_db`` feature is copied into the sample dictionary before
    ``snr_linear`` is evaluated. This preserves a simple data-flow contract for
    user-defined zero-processing features.
    """
    parameters = ParameterSet.from_dict({'snr_db': 20.0})
    config = Config(parameters=parameters)
    config.feature('snr_linear', lambda data: 10 ** (data['snr_db'] / 10))

    sample = next(Dataset(config=config).generate(features=['snr_db', 'snr_linear'], size=1, progress_bar=False))

    assert sample['snr_db'] == 20.0
    assert sample['snr_linear'] == 100.0


@parametrize_with_cases(('config_class', 'feature_names'), cases='.', prefix='case_zero_processing')
def test_shallow_parameter_paths_are_default_zero_processing_features(config_class, feature_names):
    """Test agreed shallow ParameterSet paths as zero-processing features.

    Each dataset-family config should expose its first-pass shallow parameters
    through ``Dataset.generate`` without registering explicit feature callbacks.
    Array-valued parameters are compared elementwise to avoid object-identity
    assumptions.
    """
    config = config_class()

    sample = next(Dataset(config=config).generate(features=feature_names, size=1, progress_bar=False))

    for feature_name in feature_names:
        expected = getattr(config.parameters, feature_name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(sample[feature_name], expected)
        else:
            assert sample[feature_name] == expected


@parametrize_with_cases(('config_class', 'path', 'expected'), cases='.', prefix='case_sampled_shallow')
def test_sampled_shallow_parameter_features_reflect_sampler_values(config_class, path, expected):
    """Test sampled shallow parameters are copied after samplers run.

    The feature value should reflect the sampler-updated ``ParameterSet`` state
    for direct paths such as ``snr_db``, ``rt60``, ``signal_length``, and
    ``scenario``. No processed Feature namespaces are involved in this loop.
    """
    config = config_class()
    config.sample(path, random_func=constant_sampler(expected))

    sample = next(Dataset(config=config).generate(features=[path], size=1, progress_bar=False))

    assert sample[path] == expected


def test_feature_output_names_are_not_parameter_aliases():
    """Test legacy output names are not ParameterSet aliases.

    ``source_locations`` is the canonical shallow parameter path. Requesting
    ``loc`` must fail for the new high-level ``SyntheticConfig`` until a later
    Feature retrieval layer maps legacy output names deliberately.
    """
    source_locations = np.array([[1.0], [2.0], [3.0]])
    config = SyntheticConfig(parameters={'source_locations': source_locations})

    with pytest.raises(ValueError, match='Unknown feature "loc"'):
        next(Dataset(config=config).generate(features=['loc'], size=1, progress_bar=False))


def test_prepare_outputs_are_removed_from_generated_sample():
    """Test prepare outputs are intermediate values, not generated features.

    A prepare callback may add values to the per-sample ``data`` context, but
    only requested features appear in the sample yielded by ``Dataset.generate``.
    """
    config = Config(parameters=ParameterSet.from_dict({'c': 343.0}))
    config.prepare(lambda: {'intermediate': 1.0})

    sample = next(Dataset(config=config).generate(features=['c'], size=1, progress_bar=False))

    assert set(sample) == {'idx', 'c'}


def test_unrequested_registered_feature_is_not_evaluated():
    """Test registered features run only when they are requested.

    Registering a feature with ``Config.feature`` makes it available by name,
    but neither evaluates it nor adds it to the sample unless it is listed in
    ``features=[...]``.
    """
    calls = []
    config = Config(parameters=ParameterSet.from_dict({'c': 343.0}))
    config.feature('requested', lambda: 1)
    config.feature('unrequested', lambda: calls.append('unrequested'))

    sample = next(Dataset(config=config).generate(features=['requested'], size=1, progress_bar=False))

    assert set(sample) == {'idx', 'requested'}
    assert calls == []


def test_requested_feature_can_read_prepare_data():
    """Test requested features still see prepare outputs through ``data``.

    The cleanup removes prepare outputs only after all requested features were
    evaluated, so intermediate values remain usable as feature inputs.
    """
    config = Config(parameters=ParameterSet.from_dict({'c': 343.0}))
    config.prepare(lambda parameters: {'double_c': 2 * parameters.c})
    config.feature('double_c_plus_one', lambda data: data['double_c'] + 1)

    sample = next(Dataset(config=config).generate(features=['double_c_plus_one'], size=1, progress_bar=False))

    assert sample == {'idx': 0, 'double_c_plus_one': 687.0}


def test_registering_duplicate_feature_name_raises():
    """Test that a feature name can be registered only once.

    ``Config.feature`` stores one Feature object per name, so registering the
    same name again would silently replace the first callback. Like
    ``Config.sample`` for parameter paths, it raises instead.
    """
    config = Config(parameters=ParameterSet())
    config.feature('value', lambda: 1)

    with pytest.raises(ValueError, match='Feature "value" is already registered.'):
        config.feature('value', lambda: 2)


def test_registered_feature_without_metadata_is_evaluated_but_not_a_feature_object():
    """Test registered features without ``dtype`` and ``shape``.

    Such features are generated like any other requested feature, but are not
    returned as Feature objects, since the TensorFlow paths require known
    metadata.
    """
    config = Config(parameters=ParameterSet())
    config.feature('value', lambda: 1)

    feature_func, feature_objects = config.get_features(['value'], f=None, num=0)

    assert feature_func({}) == {'value': 1}
    assert feature_objects == []


def test_feature_objects_contain_only_requested_registered_features():
    """Test feature metadata matches the requested features.

    Feature objects are used to derive TensorFlow signatures and TFRecord
    encoders, so registered features with metadata must only be included when
    they are requested and therefore present in the generated sample.
    """
    config = Config(parameters=ParameterSet())
    config.feature('requested', lambda: np.float32(1.0), dtype=np.float32, shape=())
    config.feature('unrequested', lambda: np.float32(2.0), dtype=np.float32, shape=())

    _, feature_objects = config.get_features(['requested'], f=None, num=0)

    assert [feature.name for feature in feature_objects] == ['requested']


def constant_sampler(value):
    """Return a random-function callback that always returns ``value``.

    ``AttributeSampler`` interprets one-argument callbacks as ``(rng)`` and
    two-argument callbacks as ``(rng, parameters)``. This helper keeps the cases
    explicit while avoiding accidental injection of the current ParameterSet.
    """

    def sample_value(rng):  # noqa: ARG001
        """Return the closed-over constant for one sampler invocation."""
        return value

    return sample_value
