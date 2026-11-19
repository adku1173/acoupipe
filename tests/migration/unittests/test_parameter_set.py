"""Unit tests for lightweight dataset parameter sets."""

# ruff: noqa: S101

import json

from acoupipe import datasets
from acoupipe.datasets import ParameterSet
from acoupipe.datasets.ism.parameters import SyntheticISMParameters
from acoupipe.datasets.miracle.parameters import MIRACLEParameters
from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters
from acoupipe.datasets.synthetic.micgeom import tub_vogel64_ap1
from acoupipe.datasets.synthetic.parameters import SyntheticParameters

import numpy as np
import pytest
from pytest_cases import case, parametrize_with_cases


@case(id='integer-parameter')
def case_valid_parameter_integer():
    """Return an integer parameter case."""
    return 'source_count', 1


@case(id='float-parameter')
def case_valid_parameter_float():
    """Return a float parameter case."""
    return 'c', 343.0


@case(id='contains-hyphen')
def case_invalid_name_with_hyphen():
    """Return an invalid parameter name containing a hyphen."""
    return 'not-valid'


@case(id='python-keyword')
def case_invalid_name_python_keyword():
    """Return an invalid parameter name that is a Python keyword."""
    return 'class'


@case(id='private-name')
def case_invalid_name_private():
    """Return an invalid private parameter name."""
    return '_private'


@case(id='non-string-name')
def case_invalid_name_non_string():
    """Return an invalid non-string parameter name."""
    return 1


@case(id='json')
def case_file_loader_json():
    """Return a JSON parameter file loader case."""
    return 'parameters.json', json.dumps({'source_count': 1}), 'from_json', 'source_count', 1


@case(id='toml')
def case_file_loader_toml():
    """Return a TOML parameter file loader case."""
    return 'parameters.toml', 'source_count = 2\n', 'from_toml', 'source_count', 2


@case(id='c')
def case_synthetic_default_c():
    """Return the synthetic speed-of-sound default."""
    return 'c', 343.0


@case(id='fs')
def case_synthetic_default_fs():
    """Return the synthetic sample-frequency default."""
    return 'fs', 13720.0


@case(id='signal-length')
def case_synthetic_default_signal_length():
    """Return the synthetic signal-length default."""
    return 'signal_length', 5.0


@case(id='source-locations')
def case_synthetic_default_source_locations():
    """Return the synthetic source-locations default."""
    return 'source_locations', None


@case(id='source-rms')
def case_synthetic_default_source_rms():
    """Return the synthetic source-rms default."""
    return 'source_rms', None


@case(id='snr-db')
def case_synthetic_default_snr():
    """Return the synthetic SNR default."""
    return 'snr_db', None


@case(id='num-sources')
def case_excluded_parameter_num_sources():
    """Return the derived source-count field name."""
    return 'num_sources'


@case(id='source-count')
def case_excluded_parameter_source_count():
    """Return the rejected source-count field name."""
    return 'source_count'


@case(id='legacy-nsources')
def case_excluded_parameter_nsources():
    """Return the rejected legacy source-count field name."""
    return 'nsources'


@case(id='mode')
def case_excluded_parameter_mode():
    """Return a processing-engine field name."""
    return 'mode'


@case(id='fft')
def case_excluded_parameter_fft():
    """Return a processing-engine namespace name."""
    return 'fft'


@case(id='loc')
def case_excluded_parameter_loc():
    """Return a Feature/output name that is not a ParameterSet field."""
    return 'loc'


@case(id='synthetic-ism-rt60')
def case_remaining_default_synthetic_ism_rt60():
    """Return the image-source-method reverberation-time default."""
    return SyntheticISMParameters, 'rt60', 2.0


@case(id='synthetic-ism-room-size')
def case_remaining_default_synthetic_ism_room_size():
    """Return the image-source-method room-size default."""
    return SyntheticISMParameters, 'room_size', [6, 4, 3]


@case(id='miracle-scenario')
def case_remaining_default_miracle_scenario():
    """Return the MIRACLE scenario default."""
    return MIRACLEParameters, 'scenario', 'A1'


@case(id='miracle-fs')
def case_remaining_default_miracle_fs():
    """Return the MIRACLE sampling-frequency default."""
    return MIRACLEParameters, 'fs', 32000


@case(id='miracle-signal-length')
def case_remaining_default_miracle_signal_length():
    """Return the MIRACLE signal-length default."""
    return MIRACLEParameters, 'signal_length', 5.0


@case(id='miracle-source-locations')
def case_remaining_default_miracle_source_locations():
    """Return the MIRACLE source-locations default."""
    return MIRACLEParameters, 'source_locations', None


@case(id='miracle-source-rms')
def case_remaining_default_miracle_source_rms():
    """Return the MIRACLE source-rms default."""
    return MIRACLEParameters, 'source_rms', None


@case(id='miracle-snr-db')
def case_remaining_default_miracle_snr_db():
    """Return the MIRACLE SNR default."""
    return MIRACLEParameters, 'snr_db', None


@case(id='sriracha-scenario')
def case_remaining_default_sriracha_scenario():
    """Return the SRIRACHA scenario default."""
    return SRIRACHAParameters, 'scenario', 'SR1-D'


@case(id='sriracha-fs')
def case_remaining_default_sriracha_fs():
    """Return the SRIRACHA sampling-frequency default."""
    return SRIRACHAParameters, 'fs', 32000


@parametrize_with_cases(('name', 'value'), cases='.', prefix='case_valid_parameter')
def test_parameter_set_from_dict_exposes_attribute(name, value):
    """Test that each dictionary entry is exposed as a tracked attribute."""
    assert_from_dict_exposes_attribute(name, value)


@parametrize_with_cases(('name', 'value'), cases='.', prefix='case_valid_parameter')
def test_parameter_set_constructor_exposes_attribute(name, value):
    """Test that constructor keyword arguments are exposed as attributes."""
    assert_constructor_exposes_attribute(name, value)


@parametrize_with_cases('name', cases='.', prefix='case_invalid_name')
def test_parameter_set_rejects_invalid_name(name):
    """Test that invalid Python identifiers are rejected as parameter names."""
    assert_invalid_name_is_rejected(name)


@parametrize_with_cases(
    ('file_name', 'content', 'loader_name', 'parameter_name', 'expected'), cases='.', prefix='case_file_loader'
)
def test_parameter_set_loads_file_type(tmp_path, file_name, content, loader_name, parameter_name, expected):
    """Test loading parameter sets from each supported file type."""
    assert_file_loader_reads_parameter(tmp_path, file_name, content, loader_name, parameter_name, expected)


@parametrize_with_cases(('name', 'expected'), cases='.', prefix='case_synthetic_default')
def test_synthetic_parameters_scalar_defaults(name, expected):
    """Test agreed synthetic scalar and unset defaults."""
    assert getattr(SyntheticParameters(), name) == expected


def test_synthetic_parameters_mics_default_is_legacy_geometry_copy():
    """Test that the default microphone geometry copies the legacy coordinates."""
    assert_mics_default_is_legacy_geometry_copy()


def test_synthetic_parameters_schema_keys_are_shallow_simulation_state():
    """Test that synthetic defaults contain only agreed shallow simulation parameters."""
    assert set(SyntheticParameters().to_dict()) == synthetic_schema_keys()


@parametrize_with_cases('name', cases='.', prefix='case_excluded_parameter')
def test_synthetic_parameters_exclude_derivable_and_processing_parameters(name):
    """Test that derived counts and processing-engine settings are not default fields."""
    assert_parameter_is_excluded(SyntheticParameters(), name)


def test_synthetic_parameters_are_validation_light_for_sampler_mutation():
    """Test that synthetic parameters do not validate source or microphone shapes eagerly."""
    assert_validation_light_mutation()


@parametrize_with_cases(('parameters_class', 'name', 'expected'), cases='.', prefix='case_remaining_default')
def test_remaining_parameter_classes_defaults(parameters_class, name, expected):
    """Test remaining ParameterSet class defaults."""
    assert getattr(parameters_class(), name) == expected


def test_remaining_parameter_classes_schema_keys_are_shallow_state():
    """Test remaining ParameterSet classes contain only agreed shallow state."""
    assert_remaining_schema_keys()


def test_dataset_package_exports_all_parameter_classes():
    """Test that the dataset package exports all public ParameterSet classes."""
    assert_dataset_package_exports_parameter_classes()


def test_measured_parameter_classes_list_available_scenarios():
    """Test measured ParameterSet classes expose legacy scenario vocabularies."""
    assert_measured_available_scenarios()


def assert_from_dict_exposes_attribute(name, value):
    """Assert that :meth:`ParameterSet.from_dict` tracks and exposes an attribute."""
    parameters = ParameterSet.from_dict({name: value})
    assert getattr(parameters, name) == value
    assert parameters.to_dict() == {name: value}


def assert_constructor_exposes_attribute(name, value):
    """Assert that the :class:`ParameterSet` constructor exposes an attribute."""
    assert getattr(ParameterSet(**{name: value}), name) == value


def assert_invalid_name_is_rejected(name):
    """Assert that an invalid parameter name raises :class:`ValueError`."""
    with pytest.raises(ValueError, match='Invalid parameter name'):
        ParameterSet.from_dict({name: 1})


def assert_file_loader_reads_parameter(tmp_path, file_name, content, loader_name, parameter_name, expected):
    """Assert that a file loader reads one parameter value."""
    path = tmp_path / file_name
    path.write_text(content, encoding='utf-8')
    assert getattr(getattr(ParameterSet, loader_name)(path), parameter_name) == expected


def assert_mics_default_is_legacy_geometry_copy():
    """Assert that ``mics`` defaults to a copy of the legacy geometry."""
    parameters = SyntheticParameters()
    np.testing.assert_allclose(parameters.mics, tub_vogel64_ap1)
    assert parameters.mics is not tub_vogel64_ap1


def synthetic_schema_keys():
    """Return the agreed first-pass synthetic schema keys."""
    return {'c', 'fs', 'signal_length', 'mics', 'source_locations', 'source_rms', 'snr_db'}


def assert_remaining_schema_keys():
    """Assert schema keys for the remaining ParameterSet classes."""
    assert set(SyntheticISMParameters().to_dict()) == synthetic_schema_keys() | {'rt60', 'room_size'}
    assert set(MIRACLEParameters().to_dict()) == measured_schema_keys()
    assert set(SRIRACHAParameters().to_dict()) == measured_schema_keys()


def measured_schema_keys():
    """Return agreed first-pass measured-dataset schema keys."""
    return {'scenario', 'fs', 'signal_length', 'source_locations', 'source_rms', 'snr_db'}


def assert_dataset_package_exports_parameter_classes():
    """Assert all public ParameterSet classes are available from ``acoupipe.datasets``."""
    assert datasets.SyntheticParameters is SyntheticParameters
    assert datasets.SyntheticISMParameters is SyntheticISMParameters
    assert datasets.MIRACLEParameters is MIRACLEParameters
    assert datasets.SRIRACHAParameters is SRIRACHAParameters


def assert_measured_available_scenarios():
    """Assert measured ParameterSet classes expose all legacy scenario names."""
    assert MIRACLEParameters.available_scenarios == ('A1', 'D1', 'A2', 'R2')
    assert SRIRACHAParameters.available_scenarios == (
        'SR1',
        'SR1-C1',
        'SR1-C2',
        'SR1-C3',
        'SR1-C4',
        'SR1-D',
        'SR2',
        'SR2-C1',
        'SR2-C2',
        'SR2-C3',
        'SR2-C4',
        'SR2-D',
        'SRA1',
        'SRA1-C1',
        'SRA1-C2',
        'SRA1-C3',
        'SRA1-C4',
        'SRA1-D',
        'SRA2',
        'SRA2-C1',
        'SRA2-C2',
        'SRA2-C3',
        'SRA2-C4',
        'SRA2-D',
    )


def assert_parameter_is_excluded(parameters, name):
    """Assert that a parameter name is neither tracked nor exposed."""
    assert name not in parameters.to_dict()
    assert not hasattr(parameters, name)


def assert_validation_light_mutation():
    """Assert that sampler-mutated values are stored without eager validation."""
    parameters = SyntheticParameters(source_locations='not-an-array', source_rms=object(), mics='mutable-by-sampler')
    assert parameters.source_locations == 'not-an-array'
    assert parameters.source_rms is not None
    assert parameters.mics == 'mutable-by-sampler'
