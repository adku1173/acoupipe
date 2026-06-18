# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import runpy
from pathlib import Path

import numpy as np

from acoupipe.config import TF_FLAG


EXAMPLE = Path('examples/high_level_monte_carlo_dataset.py')


def test_high_level_monte_carlo_example_is_sphinx_gallery_tutorial():
    source = EXAMPLE.read_text(encoding='utf-8')

    assert source.startswith('"""')
    assert 'High-level Monte-Carlo datasets' in source
    assert '###############################################################################' in source
    model_position = source.index('whitenoise = ac.WNoiseGenerator(')
    parameters_position = source.index('parameters = ParameterSet.from_dict(')
    assert model_position < parameters_position
    assert "'rms': 1.0" in source
    assert "'center_frequency': 1000.0" in source
    assert "'bandwidth': 1" in source
    assert "'sample_freq':" not in source
    assert "'signal_seed':" not in source
    assert "'num_samples':" not in source
    assert "'block_size':" not in source
    assert "'grid_shape':" not in source
    assert "'default':" not in source
    assert "'dtype':" not in source
    assert "'desc':" not in source
    assert 'MICROPHONE_GEOMETRY =' not in source
    assert 'CENTER_FREQUENCY =' not in source
    assert 'BANDWIDTH =' not in source
    assert 'config.sample(' in source
    assert 'random_func=' in source
    assert 'random_var=' not in source
    assert 'scipy.stats' not in source
    assert 'config.prepare(' in source
    assert 'config.feature(' in source
    assert 'dtype=np.float32' in source
    assert 'shape=grid.shape' in source
    assert 'dataset.generate(size=1' in source
    assert 'get_tensorflow_dataset' in source
    assert 'save_example_files' in source
    assert 'create_feature' not in source
    assert 'ParametersBase' not in source
    assert 'from traits.api' not in source
    assert 'run_low_level' not in source
    assert 'low_level' not in source


def test_high_level_monte_carlo_example_runs_and_returns_high_level_sample():
    namespace = runpy.run_path(EXAMPLE)

    result = namespace['run_example']()

    assert set(result) == {'rms', 'sourcemap'}
    assert result['rms'] > 0.0
    assert result['sourcemap'].shape == namespace['grid'].shape
    np.testing.assert_allclose(result['sourcemap'], result['sourcemap'].astype(np.float32), rtol=1e-6)


def test_high_level_monte_carlo_example_creates_tensorflow_dataset_when_available():
    namespace = runpy.run_path(EXAMPLE)

    if not TF_FLAG:
        return

    sample = next(iter(namespace['get_tensorflow_dataset']()))

    assert sample['rms'].shape == ()
    assert tuple(sample['sourcemap'].shape) == namespace['grid'].shape


def test_high_level_monte_carlo_example_saves_ml_ready_files(tmp_path):
    import h5py

    namespace = runpy.run_path(EXAMPLE)

    paths = namespace['save_example_files'](tmp_path)

    assert paths['h5'].exists()
    with h5py.File(paths['h5'], 'r') as h5:
        assert '1/rms' in h5
        assert '1/sourcemap' in h5
    if TF_FLAG:
        assert paths['tfrecord'].exists()
