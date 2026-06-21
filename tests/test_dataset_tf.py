"""Tests for TensorFlow dataset pipeline integration.

These tests verify that the pipeline-to-TensorFlow integration works correctly
using the real dataset, as they test the actual pipeline implementation.
"""

from acoupipe.datasets.synthetic import DatasetSynthetic

from tests.constants import FREQUENCIES, IMPLEMENTED_FEATURES, MODES, NUMS
from tests.synthetic_test_config import DatasetSyntheticTestConfig

import pytest


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
