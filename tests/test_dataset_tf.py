"""Tests for TensorFlow dataset pipeline integration.

These tests verify that the pipeline-to-TensorFlow integration works correctly
using the real dataset, as they test the actual pipeline implementation.
"""

import importlib.util

from acoupipe.datasets.synthetic import DatasetSynthetic

from tests.constants import FREQUENCIES, IMPLEMENTED_FEATURES, MODES, NUMS
from tests.synthetic_test_config import DatasetSyntheticTestConfig

import numpy as np
import pytest

TF_FLAG = importlib.util.find_spec('tensorflow') is not None


@pytest.fixture
def create_dataset():
    """Create a DatasetSynthetic instance for tests."""

    def _create_dataset(full=False, tasks=1, **kwargs):
        if full:
            return DatasetSynthetic(tasks=tasks, **kwargs)
        config = DatasetSyntheticTestConfig(**kwargs)
        return DatasetSynthetic(config=config, tasks=tasks, **kwargs)

    return _create_dataset


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    import shutil
    import tempfile
    from pathlib import Path

    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', FREQUENCIES)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
def test_get_tf_dataset(mode, feature, num, f, mic_sig_noise, create_dataset, temp_dir):
    """Test if a TensorFlow dataset can be constructed from the pipeline.

    This test uses the real dataset because it tests pipeline-to-TensorFlow integration,
    which depends on the actual pipeline implementation.
    """
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


@pytest.mark.skipif(not TF_FLAG, reason='TensorFlow not available')
@pytest.mark.parametrize(
    ('feature', 'kwargs'),
    [
        ('csm', {'mode': 'analytic'}),  # fixed shape, complex-valued
        ('loc', {'mode': 'analytic'}),  # one varying dimension
        ('time_data', {'mode': 'welch', 'random_signal_length': True, 'signal_length': 0.1}),  # two varying dimensions
    ],
)
def test_save_tfrecord_roundtrip(feature, kwargs, create_dataset, temp_dir):
    """Test that samples written by save_tfrecord are restored correctly by get_tfrecord_parser."""
    import tensorflow as tf

    dataset = create_dataset(**kwargs)
    size = 2
    generated = list(
        dataset.generate(features=[feature], f=1000, num=0, split='training', size=size, progress_bar=False)
    )
    name = temp_dir / 'test.tfrecord'
    dataset.save_tfrecord(features=[feature], f=1000, num=0, split='training', size=size, name=name, progress_bar=False)

    parser = dataset.get_tfrecord_parser(features=[feature], f=1000, num=0)
    loaded = list(tf.data.TFRecordDataset(name.as_posix()).map(parser))

    assert len(loaded) == size
    loaded = sorted(loaded, key=lambda d: int(d['idx']))
    for data_generated, data_loaded in zip(generated, loaded, strict=True):
        assert int(data_loaded['idx']) == data_generated['idx']
        np.testing.assert_array_equal(data_loaded['seeds'].numpy(), data_generated['seeds'])
        np.testing.assert_allclose(data_loaded[feature].numpy(), data_generated[feature], rtol=1e-5, atol=1e-8)
