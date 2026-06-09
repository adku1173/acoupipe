"""Tests for the dummy dataset functionality."""

import os
import importlib.util
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# Only run these tests if TensorFlow is available
TF_FLAG = importlib.util.find_spec('tensorflow') is not None

from .dummy_dataset import DatasetDummy, DatasetDummyConfig


@pytest.fixture
def dummy_dataset():
    """Create a DatasetDummy instance for tests."""
    return DatasetDummy(mode='welch', mic_sig_noise=True, use_cache=False)


@pytest.fixture
def temp_cache_dir():
    """Create and clean up a temporary cache directory."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


class TestDatasetDummyConfig:
    """Tests for DatasetDummyConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetDummyConfig()
        assert config.mode == 'welch'
        assert config.mic_sig_noise is True
        assert config.fs == 13720.0
        assert config.signal_length == 5.0
        assert config.block_size == 128
        assert config.num_mics == 64

    def test_feature_shapes(self):
        """Test that feature shapes are defined."""
        config = DatasetDummyConfig()
        assert 'csm' in config.feature_shapes
        assert 'sourcemap' in config.feature_shapes
        assert 'loc' in config.feature_shapes

    def test_feature_dtypes(self):
        """Test that feature dtypes are defined."""
        config = DatasetDummyConfig()
        assert 'csm' in config.feature_dtypes
        assert 'sourcemap' in config.feature_dtypes
        assert 'loc' in config.feature_dtypes


class TestDatasetDummy:
    """Tests for DatasetDummy."""

    def test_basic_generation(self, dummy_dataset):
        """Test basic data generation."""
        features = ['csm', 'loc', 'idx', 'seeds']
        gen = dummy_dataset.generate(features=features, size=1, split='training', f=1000, num=0, progress_bar=False)

        data = next(gen)

        assert 'idx' in data
        assert 'seeds' in data
        assert 'csm' in data
        assert 'loc' in data

        assert data['idx'] == 0
        assert isinstance(data['seeds'], np.ndarray)
        assert len(data['seeds']) == 4

    def test_feature_shapes(self, dummy_dataset):
        """Test that generated features have correct shapes."""
        config = dummy_dataset.config

        # Test various feature/parameter combinations
        test_cases = [
            (['csm'], 1000, 0),
            (['sourcemap'], 1000, 0),
            (['loc'], 1000, 0),
            (['f'], 1000, 0),
            (['num'], 1000, 3),
        ]

        for features, f, num in test_cases:
            gen = dummy_dataset.generate(
                features=features + ['idx', 'seeds'], size=1, split='training', f=f, num=num, progress_bar=False
            )

            data = next(gen)
            feature_name = features[0]

            # Check that feature exists
            assert feature_name in data, f'Feature {feature_name} not in data'

            # Check shape if it's an array
            if hasattr(data[feature_name], 'shape'):
                expected_shape = config._get_feature_shape(feature_name, f, num)
                actual_shape = data[feature_name].shape

                # Compare non-None dimensions
                for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                    if expected is not None:
                        assert expected == actual, (
                            f'Shape mismatch for {feature_name}: expected {expected_shape}, got {actual_shape}'
                        )

    def test_feature_dtypes(self, dummy_dataset):
        """Test that generated features have correct dtypes."""
        config = dummy_dataset.config

        test_cases = [
            ('csm', np.complex64),
            ('sourcemap', np.float32),
            ('loc', np.float32),
            ('f', np.float32),
            ('idx', np.int64),
            ('seeds', np.int64),
        ]

        for feature_name, expected_dtype in test_cases:
            gen = dummy_dataset.generate(
                features=[feature_name, 'idx', 'seeds'], size=1, split='training', f=1000, num=0, progress_bar=False
            )

            data = next(gen)
            actual_dtype = data[feature_name].dtype

            assert actual_dtype == expected_dtype, (
                f'Dtype mismatch for {feature_name}: expected {expected_dtype}, got {actual_dtype}'
            )

    def test_multiple_samples(self, dummy_dataset):
        """Test generating multiple samples."""
        features = ['csm', 'idx', 'seeds']
        size = 5

        gen = dummy_dataset.generate(features=features, size=size, split='training', f=1000, num=0, progress_bar=False)

        samples = list(gen)
        assert len(samples) == size

        # Check that indices are correct
        for i, sample in enumerate(samples):
            assert sample['idx'] == i

    def test_caching(self, temp_cache_dir):
        """Test that caching works correctly."""
        # Create dataset with caching
        dataset = DatasetDummy(mode='welch', cache_dir=temp_cache_dir, use_cache=True, regenerate_cache=False)

        features = ['csm', 'idx', 'seeds']

        # First generation should miss cache
        gen1 = dataset.generate(
            features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        data1 = next(gen1)

        stats1 = dataset.get_cache_stats()
        assert stats1['misses'] == 1
        assert stats1['hits'] == 0

        # Second generation with same parameters should hit cache
        dataset2 = DatasetDummy(mode='welch', cache_dir=temp_cache_dir, use_cache=True, regenerate_cache=False)

        gen2 = dataset2.generate(
            features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        data2 = next(gen2)

        stats2 = dataset2.get_cache_stats()
        assert stats2['hits'] == 1
        assert stats2['misses'] == 0

        # Data should be the same
        np.testing.assert_array_equal(data1['csm'], data2['csm'])

    def test_cache_regeneration(self, temp_cache_dir):
        """Test cache regeneration."""
        dataset = DatasetDummy(
            mode='welch',
            cache_dir=temp_cache_dir,
            use_cache=True,
            regenerate_cache=True,  # Force regeneration
        )

        features = ['csm', 'idx', 'seeds']

        # Generate data
        gen = dataset.generate(
            features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        data = next(gen)

        # Stats should show miss (since we're regenerating)
        stats = dataset.get_cache_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0

    def test_no_cache(self, temp_cache_dir):
        """Test with caching disabled."""
        dataset = DatasetDummy(mode='welch', cache_dir=temp_cache_dir, use_cache=False)

        features = ['csm', 'idx', 'seeds']

        # Generate data multiple times
        for _ in range(3):
            gen = dataset.generate(
                features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
            )
            next(gen)

        # Stats should show no cache usage
        stats = dataset.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_clear_cache(self, temp_cache_dir):
        """Test clearing the cache."""
        dataset = DatasetDummy(mode='welch', cache_dir=temp_cache_dir, use_cache=True)

        features = ['csm', 'idx', 'seeds']

        # Generate and cache data
        gen = dataset.generate(
            features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        next(gen)

        # Clear cache
        dataset.clear_cache()

        # Next generation should miss cache
        gen2 = dataset.generate(
            features=features, size=1, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        next(gen2)

        stats = dataset.get_cache_stats()
        assert stats['misses'] == 1

    def test_different_modes(self):
        """Test that different modes work."""
        for mode in ['welch', 'analytic', 'wishart']:
            dataset = DatasetDummy(mode=mode)

            gen = dataset.generate(
                features=['csm', 'idx', 'seeds'], size=1, split='training', f=1000, num=0, progress_bar=False
            )

            data = next(gen)
            assert 'csm' in data
            assert 'idx' in data

    def test_different_parameters(self, dummy_dataset):
        """Test different parameter combinations."""
        parameter_combinations = [
            {'f': 1000, 'num': 0},
            {'f': 2000, 'num': 0},
            {'f': [1000, 2000], 'num': 0},
            {'f': None, 'num': 0},
            {'f': 1000, 'num': 3},
        ]

        for params in parameter_combinations:
            gen = dummy_dataset.generate(
                features=['csm', 'idx', 'seeds'], size=1, split='training', progress_bar=False, **params
            )

            data = next(gen)
            assert 'csm' in data
            assert data['idx'] == 0


class TestDummyDatasetWithTFRecord:
    """Test that dummy dataset works with TFRecord functionality."""

    @pytest.mark.skipif(not TF_FLAG, reason='TensorFlow not available')
    def test_save_and_load_tfrecord(self, dummy_dataset, temp_dir):
        """Test saving and loading TFRecord files with dummy dataset."""
        if TF_FLAG:
            import tensorflow as tf
        else:
            pytest.skip('TensorFlow not available')

        features = ['csm', 'idx', 'seeds']

        # Save to TFRecord
        dummy_dataset.save_tfrecord(
            features=features,
            size=2,
            name=temp_dir / 'test.tfrecord',
            split='training',
            f=1000,
            num=0,
            progress_bar=False,
        )

        # Load and parse
        parser = dummy_dataset.get_tfrecord_parser(features, f=1000, num=0)
        tfrecord_dataset = tf.data.TFRecordDataset(temp_dir / 'test.tfrecord').map(parser)

        # Check that we can parse the data
        data_list = list(tfrecord_dataset)
        assert len(data_list) == 2

        for data in data_list:
            assert 'csm' in data
            assert 'idx' in data
            assert 'seeds' in data

    @pytest.mark.skipif(not bool(int(os.environ.get('TF_FLAG', 0))), reason='TensorFlow not available')
    def test_get_tf_dataset(self, dummy_dataset):
        """Test creating a TensorFlow dataset from dummy data."""
        if not TF_FLAG:
            pytest.skip('TensorFlow not available')
        import tensorflow as tf

        features = ['csm', 'idx', 'seeds']

        tf_dataset = dummy_dataset.get_tf_dataset(
            features=features, size=3, split='training', f=1000, num=0, progress_bar=False
        )

        # Check that we can iterate over the dataset
        data_list = list(tf_dataset)
        assert len(data_list) == 3

        for data in data_list:
            assert 'csm' in data
            assert 'idx' in data
            assert 'seeds' in data


class TestDummyDatasetWithHDF5:
    """Test that dummy dataset works with HDF5 functionality."""

    def test_save_h5(self, dummy_dataset, temp_dir):
        """Test saving HDF5 files with dummy dataset."""
        import h5py

        features = ['csm', 'idx', 'seeds']

        # Save to HDF5
        dummy_dataset.save_h5(
            features=features, size=2, name=temp_dir / 'test.h5', split='training', f=1000, num=0, progress_bar=False
        )

        # Check that file was created
        h5_path = temp_dir / 'test.h5'
        assert h5_path.exists()

        # Load and verify data
        with h5py.File(h5_path, 'r') as f:
            # Check that samples are stored as groups
            assert '0' in f
            assert '1' in f

            # Check features in first sample
            for feature in features:
                assert feature in f['0']

            # Verify idx values
            assert f['0/idx'][()] == 0
            assert f['1/idx'][()] == 1

    def test_save_and_load_h5(self, dummy_dataset, temp_dir):
        """Test saving and loading HDF5 files with dummy dataset."""
        import h5py
        import numpy as np

        features = ['csm', 'loc', 'idx', 'seeds']

        # First generate data to compare
        gen = dummy_dataset.generate(
            features=features, size=2, split='training', f=1000, num=0, start_idx=0, progress_bar=False
        )
        generated_data = list(gen)

        # Save to HDF5
        dummy_dataset.save_h5(
            features=features,
            size=2,
            name=temp_dir / 'test.h5',
            split='training',
            f=1000,
            num=0,
            start_idx=0,
            progress_bar=False,
        )

        # Load and compare
        h5_path = temp_dir / 'test.h5'
        with h5py.File(h5_path, 'r') as f:
            for idx, gen_sample in enumerate(generated_data):
                group_name = str(idx)
                assert group_name in f

                for feature in features:
                    assert feature in f[group_name]
                    loaded_data = f[group_name][feature][()]

                    # For complex arrays, we need to handle the conversion
                    if np.iscomplexobj(gen_sample[feature]):
                        # HDF5 stores complex as separate real/imag or as compound type
                        # Let's just check shape matches
                        assert loaded_data.shape == gen_sample[feature].shape
                    else:
                        np.testing.assert_array_equal(loaded_data, gen_sample[feature])

    def test_save_h5_with_different_features(self, dummy_dataset, temp_dir):
        """Test saving HDF5 with various feature types."""
        import h5py

        # Test with different feature combinations
        feature_sets = [
            ['csm', 'idx'],
            ['sourcemap', 'loc', 'idx'],
            ['f', 'num', 'idx'],
            ['time_data', 'idx'],
        ]

        for features in feature_sets:
            h5_path = temp_dir / f'test_{len(features)}.h5'

            # Save to HDF5
            dummy_dataset.save_h5(
                features=features, size=1, name=h5_path, split='training', f=1000, num=0, progress_bar=False
            )

            # Verify file was created
            assert h5_path.exists()

            # Check features are present
            with h5py.File(h5_path, 'r') as f:
                assert '0' in f
                for feature in features + ['idx', 'seeds']:
                    assert feature in f['0'], f'Feature {feature} not found in HDF5 file for feature set {features}'
