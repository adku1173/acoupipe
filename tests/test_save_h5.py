"""Tests for HDF5 file saving functionality using dummy dataset.

These tests verify that HDF5 file saving works correctly, decoupled from
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
@pytest.mark.parametrize('num', [0, 3])
@pytest.mark.parametrize('f', [None, 1000])
def test_save_h5(feature, num, f, dummy_dataset, temp_dir):
    """Test saving data to HDF5 format using dummy dataset."""
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')
    if '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')

    dummy_dataset.save_h5(
        split='training', num=num, f=f, size=2, features=[feature], name=temp_dir / 'test.h5', progress_bar=False
    )


@pytest.mark.parametrize('feature', ['csm', 'f'])
def test_save_and_load_h5(feature, dummy_dataset, temp_dir):
    """Test saving and loading HDF5 files with dummy dataset."""
    import h5py

    # Generate and save
    dummy_dataset.save_h5(
        features=[feature], size=2, name=temp_dir / 'test.h5', split='training', f=1000, num=0, progress_bar=False
    )

    # Load and verify
    h5_path = temp_dir / 'test.h5'
    with h5py.File(h5_path, 'r') as f:
        assert '0' in f
        assert '1' in f
        assert feature in f['0']
