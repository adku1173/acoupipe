"""Tests for TFRecord file saving functionality using dummy dataset.

These tests verify that TFRecord file saving works correctly, decoupled from
acoustic correctness testing (which is done in regression tests).
"""

import numpy as np
import pytest

from .dummy_dataset import DatasetDummy


@pytest.fixture
def dummy_dataset():
    """Create a DatasetDummy instance for fast I/O testing."""
    return DatasetDummy(mode='analytic', use_cache=False)


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    import tempfile
    import shutil
    from pathlib import Path

    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.mark.parametrize('feature', ['csm', 'f'])
@pytest.mark.parametrize('mic_sig_noise', [True, False])
def test_save_tfrecord(feature, mic_sig_noise, dummy_dataset, temp_dir):
    """Test saving data to TFRecord format using dummy dataset."""
    if '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')

    dummy_dataset.save_tfrecord(
        split='training', size=2, features=[feature], name=temp_dir / 'test.tfrecord', progress_bar=False
    )


@pytest.mark.parametrize('feature', ['csm', 'f'])
@pytest.mark.parametrize('mic_sig_noise', [True, False])
def test_parse_tfrecord(feature, mic_sig_noise, dummy_dataset, temp_dir):
    """Test parsing TFRecord files using dummy dataset."""
    import tensorflow as tf

    if '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')

    # Generate data
    data_generated = next(
        dummy_dataset.generate(f=1000, num=0, split='training', size=1, progress_bar=False, features=[feature])
    )

    # Save and parse
    dummy_dataset.save_tfrecord(
        f=1000, num=0, split='training', size=1, progress_bar=False, features=[feature], name=temp_dir / 'test.tfrecord'
    )

    parser = dummy_dataset.get_tfrecord_parser(f=1000, num=0, features=[feature])
    tfrecord = tf.data.TFRecordDataset((temp_dir / 'test.tfrecord').as_posix()).map(parser)
    data_loaded = next(iter(tfrecord))

    # Compare data
    np.testing.assert_allclose(data_generated[feature], data_loaded[feature], rtol=1e-5, atol=1e-8)
