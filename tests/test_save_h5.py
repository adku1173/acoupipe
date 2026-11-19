"""Tests for HDF5 file saving functionality using dummy dataset.

These tests verify that HDF5 file saving works correctly, decoupled from
acoustic correctness testing (which is done in regression tests).
"""

from acoupipe import writer as writer_module
from acoupipe.loader import LoadH5Dataset
from acoupipe.pipeline import BasePipeline
from acoupipe.writer import WriteH5Dataset

from .dummy_dataset import DatasetDummy

import numpy as np
import pytest
from h5py import File as H5File


@pytest.fixture
def dummy_dataset():
    """Create a DatasetDummy instance for fast I/O testing."""
    return DatasetDummy(mode='analytic', use_cache=False)


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    import shutil
    import tempfile
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


@pytest.mark.parametrize(
    ('method', 'outcome'),
    [
        ('save', 'done'),
        ('save', 'error'),
        ('get_data', 'done'),
        ('get_data', 'error'),
        ('get_data', 'close'),
        ('get_data', 'stop'),
    ],
)
def test_writer_closes_file(method, outcome, tmp_path, monkeypatch):
    """Close the real file on completion, source errors and stream cancellation."""
    opened = []

    def open_file(*args, **kwargs):
        file = H5File(*args, **kwargs)
        opened.append(file)
        return file

    def calculate(_sampler):
        if outcome == 'error':
            msg = 'source failed'
            raise ValueError(msg)
        return {'value': 1}

    monkeypatch.setattr(writer_module, 'H5File', open_file)
    pipeline = BasePipeline(numsamples=2, feature_func=calculate)
    writer = WriteH5Dataset(source=pipeline, name=str(tmp_path / 'close.h5'))

    def run():
        if method == 'save':
            writer.save(progress_bar=False)
        else:
            list(writer.get_data(progress_bar=False))

    if outcome == 'error':
        with pytest.raises(ValueError, match='source failed'):
            run()
    elif outcome == 'done':
        run()
    else:
        stream = writer.get_data(progress_bar=False)
        next(stream)
        if outcome == 'close':
            stream.close()
        else:
            writer.writeflag = False
            list(stream)
    assert len(opened) == 1
    assert not opened[0].id.valid


def assert_nested_equal(actual, expected):
    """Compare nested dictionaries, including array shapes and dtypes."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_nested_equal(actual[key], value)
    else:
        np.testing.assert_array_equal(actual, expected, strict=True)


@pytest.mark.parametrize('write_method', ['save', 'get_data'])
@pytest.mark.parametrize('write_features', [[], ['method']])
@pytest.mark.parametrize(
    ('read_method', 'read_features'),
    [('get_data', None), ('get_dataset_generator', None), ('get_dataset_generator', ['method'])],
)
def test_nested_h5_roundtrip(write_method, write_features, read_method, read_features, tmp_path):
    """Round-trip nested samples and metadata through both writers and readers."""
    samples = [
        {
            'flat': np.array([1 + 2j, 3], dtype=np.complex64),
            'method': {
                'sourcemap': np.arange(6, dtype=np.float32).reshape(2, 3),
                'time': 0.125,
                'specific_level_error': np.zeros((2, nsources)),
                'solver_info': {'iterations': np.array([2, 3], dtype=np.int16), 'converged': True},
                'empty': {},
            },
        }
        for nsources in (1, 2)
    ]
    values = iter(samples)
    pipeline = BasePipeline(numsamples=2, feature_func=lambda _sampler: next(values))
    metadata = {'version': 1, 'solver': {'tolerance': 1e-6, 'empty': {}}}
    filename = str(tmp_path / 'nested.h5')
    writer = WriteH5Dataset(source=pipeline, name=filename, features=write_features, metadata=metadata)
    if write_method == 'save':
        writer.save(progress_bar=False)
    else:
        list(writer.get_data(progress_bar=False))

    loader = LoadH5Dataset(name=filename)
    with loader.h5f:
        assert loader.numsamples == 2
        assert set(loader.features) == ({'idx', 'method'} if write_features else {'idx', 'flat', 'method'})
        assert_nested_equal(loader.metadata, metadata)
        if read_method == 'get_data':
            loaded = list(loader.get_data())
        else:
            loaded = list(loader.get_dataset_generator(features=read_features)())
        assert len(loaded) == 2
        for index, (actual, sample) in enumerate(zip(loaded, samples, strict=True), start=1):
            expected = {'idx': str(index), 'method': sample['method']}
            if not write_features and read_features is None:
                expected['flat'] = sample['flat']
            assert_nested_equal(actual, expected)
