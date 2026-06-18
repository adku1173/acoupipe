# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import runpy
from pathlib import Path

import numpy as np

from acoupipe.config import TF_FLAG


INTRO_EXAMPLE = Path('examples/high_level_monte_carlo_dataset.py')
FEATURE_EXAMPLE = Path('examples/high_level_dataset_features.py')


def test_introductory_high_level_monte_carlo_example_remains_available():
    source = INTRO_EXAMPLE.read_text(encoding='utf-8')

    assert 'High-level Monte-Carlo datasets with the Dataset API' in source
    assert 'config.feature(' in source
    assert 'save_example_files' in source


def test_high_level_dataset_features_example_is_follow_up_gallery_tutorial():
    source = FEATURE_EXAMPLE.read_text(encoding='utf-8')

    assert source.startswith('"""')
    assert 'Adding Features to high-level Monte-Carlo Datasets' in source
    assert 'examples/high_level_monte_carlo_dataset.py' in source
    assert '###############################################################################' in source
    setup_position = source.index('whitenoise = ac.WNoiseGenerator(')
    callback_position = source.index('config.feature(')
    wrapped_position = source.index('rms_db_feature = create_feature(')
    catalog_position = source.index('class NormalizedSourcemapFeature')
    default_position = source.index('def _get_default_feature_rms_ratio')
    assert setup_position < callback_position < wrapped_position < catalog_position < default_position
    assert 'TensorFlow' in source
    assert 'TFRecord' in source
    assert 'HDF5' in source
    assert 'Feature object' in source
    assert 'Feature' in source
    assert 'ParameterSet' in source
    assert 'get_prepare_func' in source
    assert 'prepare_normalized_sourcemap' in source


def test_high_level_dataset_features_example_runs_all_four_feature_paths():
    namespace = runpy.run_path(FEATURE_EXAMPLE)

    sample = namespace['run_example']()
    expected_rms = np.float32(np.random.default_rng(1).rayleigh(scale=5.0))

    np.testing.assert_allclose(sample['rms'], expected_rms, rtol=1e-6)
    np.testing.assert_allclose(sample['rms_db'], np.float32(20.0 * np.log10(expected_rms)), rtol=1e-6)
    np.testing.assert_allclose(
        sample['rms_ratio'], np.float32(expected_rms / namespace['parameters'].reference_rms), rtol=1e-6
    )
    assert sample['normalized_sourcemap'].shape == namespace['grid'].shape
    np.testing.assert_allclose(sample['normalized_sourcemap'].max(), np.float32(1.0), rtol=1e-6)
    assert namespace['normalized_sourcemap_feature'].beamformer.steer.env.c == namespace['parameters'].sourcemap.c


def test_high_level_dataset_features_example_saves_all_features_to_h5(tmp_path):
    import h5py

    namespace = runpy.run_path(FEATURE_EXAMPLE)

    path = namespace['save_h5_with_all_features'](tmp_path)

    assert path.exists()
    with h5py.File(path, 'r') as h5:
        for feature in ['rms', 'rms_db', 'normalized_sourcemap', 'rms_ratio']:
            assert f'1/{feature}' in h5


def test_high_level_dataset_features_example_tfrecord_save_accepts_all_feature_objects(tmp_path):
    namespace = runpy.run_path(FEATURE_EXAMPLE)

    if not TF_FLAG:
        return

    path = namespace['save_tfrecord_with_all_features'](tmp_path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_high_level_dataset_features_example_tensorflow_dataset_uses_string_addressable_features():
    namespace = runpy.run_path(FEATURE_EXAMPLE)

    if not TF_FLAG:
        return

    sample = next(iter(namespace['get_tensorflow_dataset_with_string_features']()))
    expected_rms = np.float32(np.random.default_rng(1).rayleigh(scale=5.0))

    assert sample['rms'].shape == ()
    assert sample['rms_ratio'].shape == ()
    np.testing.assert_allclose(sample['rms'].numpy(), expected_rms, rtol=1e-6)
    np.testing.assert_allclose(
        sample['rms_ratio'].numpy(),
        np.float32(expected_rms / namespace['parameters'].reference_rms),
        rtol=1e-6,
    )


def test_high_level_dataset_features_example_documents_direct_object_tf_dataset_limitation():
    namespace = runpy.run_path(FEATURE_EXAMPLE)

    if not TF_FLAG:
        return

    result = namespace['try_tensorflow_dataset_with_direct_feature_object']()

    assert result['compatible'] is False
    assert result['api'] == 'get_tf_dataset'
    assert 'Feature object' in result['reason']
