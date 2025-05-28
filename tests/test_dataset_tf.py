import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from acoupipe.datasets.synthetic import DatasetSynthetic, DatasetSyntheticTestConfig

IMPLEMENTED_FEATURES = ['time_data', 'csm', 'csmtriu', 'sourcemap', 'eigmode', 'spectrogram'] + [
    'seeds',
    'idx',
    'loc',
    'source_strength_analytic',
    'source_strength_estimated',
    'noise_strength_analytic',
    'noise_strength_estimated',
    'f',
    'num',
    'targetmap_analytic',
    'targetmap_estimated',
]
modes = ['welch', 'analytic', 'wishart']
frequencies = [None, 1000]
nums = [0, 3]
start_idx = 3


@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def create_dataset():
    """Fixture to create a DatasetSynthetic instance."""

    def _create_dataset(full=False, tasks=1, **kwargs):
        if full:
            return DatasetSynthetic(tasks=tasks, **kwargs)
        config = DatasetSyntheticTestConfig(**kwargs)
        return DatasetSynthetic(config=config, tasks=tasks, **kwargs)

    return _create_dataset


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES + ['idx', 'seeds'])
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
def test_parse_tfrecord(mode, feature, mic_sig_noise, num, f, temp_dir, create_dataset):
    """Test parsing TFRecord files."""
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')
    if mode == 'analytic' and 'estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    # generate data
    dataset = create_dataset(mode=mode, mic_sig_noise=mic_sig_noise)
    data_generated = next(
        dataset.generate(f=f, num=num, split='training', size=1, progress_bar=False, features=[feature])
    )
    # save and parse data
    dataset.save_tfrecord(
        f=f, num=num, split='training', size=1, progress_bar=False, features=[feature], name=temp_dir / 'test.tfrecord'
    )
    parser = dataset.get_tfrecord_parser(f=f, num=num, features=[feature])
    tfrecord = tf.data.TFRecordDataset(temp_dir / 'test.tfrecord').map(parser)
    data_loaded = next(iter(tfrecord))
    # compare data
    np.testing.assert_allclose(data_generated[feature], data_loaded[feature], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
def test_save_tfrecord(mode, feature, mic_sig_noise, temp_dir, create_dataset):
    """Test saving data to TFRecord format."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode, mic_sig_noise=mic_sig_noise)
    dataset.save_tfrecord(
        split='training', size=2, features=[feature], name=temp_dir / 'test.tfrecord', progress_bar=False
    )


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
def test_get_tf_dataset(mode, feature, num, f, mic_sig_noise, create_dataset):
    """Test if a TensorFlow dataset can be constructed from the pipeline."""
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode, mic_sig_noise=mic_sig_noise)
    dataset = dataset.get_tf_dataset(split='training', size=1, progress_bar=False, f=f, num=num, features=[feature])
    data = next(iter(dataset))
    assert feature in data.keys()
