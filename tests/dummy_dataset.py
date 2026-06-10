"""Dummy dataset for fast testing without expensive acoustic calculations.

This module provides a DatasetDummy class that generates synthetic data with
correct shapes and data types, caches it to disk, and loads from cache on
subsequent runs. This is useful for testing TFRecord serialization and other
I/O operations without triggering the computationally expensive acoustic
pipeline.
"""

import hashlib
import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
from traits.api import Bool, Dict, Float, Instance, Int, List, Enum, Any

from acoupipe.datasets.base import ConfigBase, DatasetBase
from acoupipe.datasets.features import BaseFeatureCatalog, create_feature


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'acoupipe' / 'dummy_dataset'


class DatasetDummyConfig(ConfigBase):
    """Configuration for the dummy dataset.

    This config provides minimal setup without creating heavy acoular objects,
    and defines the data shapes and types for dummy features.
    """

    # Mode (matches real dataset modes for compatibility)
    mode = Enum(('welch', 'analytic', 'wishart'), default='welch', desc='type of calculation method')

    # Noise configuration
    mic_sig_noise = Bool(True, desc='apply signal noise to microphone signals')

    # FFT parameters (simplified)
    fs = Float(13720.0, desc='sampling frequency')
    signal_length = Float(5.0, desc='signal length in seconds')
    block_size = Int(128, desc='FFT block size')
    nfft = Int(desc='Number of frequency bins')
    num_time_bins = Int(desc='Number of time bins for spectrogram')

    # Array parameters
    num_mics = Int(64, desc='number of microphones')
    num_grid_points = Int(1024, desc='number of grid points for beamforming')

    # Feature shape mappings: (f, num, mode) -> shape
    # These are simplified but match the real dataset shapes
    feature_shapes = Dict(
        {
            'time_data': (None, None),  # (time_samples, num_mics)
            'csm': (None, None, None),  # (freq_bins, num_mics, num_mics)
            'csmtriu': (None, None, None),  # (freq_bins, num_mics, num_mics)
            'sourcemap': (None, None),  # (freq_bins, grid_points)
            'eigmode': (None, None, None),  # (freq_bins, num_mics, num_mics)
            'spectrogram': (None, None, None),  # (freq_bins, time_bins, num_mics)
            'loc': (3, None),  # (3, num_sources)
            'source_strength_analytic': (None, None),  # (freq_bins, num_sources)
            'source_strength_estimated': (None, None),  # (freq_bins, num_sources)
            'noise_strength_analytic': (None, None),  # (freq_bins, num_mics)
            'noise_strength_estimated': (None, None),  # (freq_bins, num_mics)
            'f': (None,),  # (freq_bins,)
            'num': (),  # scalar
            'targetmap_analytic': (None, None),  # (freq_bins, grid_points)
            'targetmap_estimated': (None, None),  # (freq_bins, grid_points)
            'seeds': (None,),  # variable length
            'idx': (),  # scalar
        }
    )

    # Feature dtype mappings
    feature_dtypes = Dict(
        {
            'time_data': np.float32,
            'csm': np.complex64,
            'csmtriu': np.float32,
            'sourcemap': np.float32,
            'eigmode': np.complex64,
            'spectrogram': np.complex64,
            'loc': np.float32,
            'source_strength_analytic': np.float32,
            'source_strength_estimated': np.float32,
            'noise_strength_analytic': np.float32,
            'noise_strength_estimated': np.float32,
            'f': np.float32,
            'num': np.int64,
            'targetmap_analytic': np.float32,
            'targetmap_estimated': np.float32,
            'seeds': np.int64,
            'idx': np.int64,
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Calculate derived parameters
        self.nfft = self.block_size // 2 + 1  # Number of frequency bins
        self.num_time_bins = int(self.signal_length * self.fs / (self.block_size / 2)) - 1  # For spectrogram

    def get_sampler(self):
        """Return empty sampler dict since we don't need real sampling."""
        return {}

    def get_default_features(self, features, f, num):
        """Build dummy feature catalog objects."""
        default_features = []
        for feature_name in features:
            if feature_name in ['idx', 'seeds']:
                continue
            default_features.append(self._get_dummy_feature(feature_name, f, num))
        return default_features

    def _get_dummy_feature(self, feature_name, f, num):
        """Create a dummy feature with correct shape and dtype."""
        shape = self._get_feature_shape(feature_name, f, num)
        dtype = self.feature_dtypes[feature_name]

        if feature_name == 'f':
            # Frequency array
            if f is None:
                f_values = np.linspace(100, 10000, self.nfft, dtype=np.float32)
            elif isinstance(f, list):
                f_values = np.array(f, dtype=np.float32)
            else:
                f_values = np.array([f], dtype=np.float32)
            return create_feature(
                feature_func=lambda sampler: {'f': f_values},
                name='f',
                shape=f_values.shape,
                dtype=self.feature_dtypes['f'],  # Use the type, not the dtype instance
            )
        elif feature_name == 'num':
            return create_feature(
                feature_func=lambda sampler: {'num': num},
                name='num',
                shape=(),
                dtype=np.int64,
            )
        else:
            # For other features, create a callable that returns random data
            def feature_func(sampler, name=feature_name, shape=shape, dtype=dtype):
                data = self._generate_dummy_array(shape, dtype, name)
                return {name: data}

            return create_feature(
                feature_func=feature_func,
                name=feature_name,
                shape=shape,
                dtype=dtype,
            )

    def _get_feature_shape(self, feature_name, f, num):
        """Get the shape for a feature based on parameters."""
        # Handle frequency dimension
        if f is None:
            freq_dim = self.nfft
        elif isinstance(f, list):
            freq_dim = len(f)
        else:
            freq_dim = 1

        # Handle num dimension (fractional octave bands)
        if num == 0:
            num_dim = 1
        else:
            num_dim = num + 1  # Simplified

        # Fixed dimensions from config
        num_mics = self.num_mics
        num_grid_points = self.num_grid_points
        signal_length = self.signal_length
        fs = self.fs

        # Define shapes for each feature
        if feature_name == 'time_data':
            return (int(signal_length * fs), num_mics)
        elif feature_name == 'csm':
            return (freq_dim * num_dim, num_mics, num_mics)
        elif feature_name == 'csmtriu':
            return (freq_dim * num_dim, num_mics, num_mics)
        elif feature_name == 'sourcemap':
            return (freq_dim * num_dim, num_grid_points)
        elif feature_name == 'eigmode':
            return (freq_dim * num_dim, num_mics, num_mics)
        elif feature_name == 'spectrogram':
            return (freq_dim * num_dim, self.num_time_bins, num_mics)
        elif feature_name == 'loc':
            return (3, 3)  # (3, num_sources) with fixed 3 sources
        elif feature_name == 'source_strength_analytic':
            return (freq_dim * num_dim, 3)  # (freq_bins, num_sources)
        elif feature_name == 'source_strength_estimated':
            return (freq_dim * num_dim, 3)  # (freq_bins, num_sources)
        elif feature_name == 'noise_strength_analytic':
            return (freq_dim * num_dim, num_mics)
        elif feature_name == 'noise_strength_estimated':
            return (freq_dim * num_dim, num_mics)
        elif feature_name == 'f':
            return (freq_dim,)
        elif feature_name == 'num':
            return ()
        elif feature_name == 'targetmap_analytic':
            return (freq_dim * num_dim, num_grid_points)
        elif feature_name == 'targetmap_estimated':
            return (freq_dim * num_dim, num_grid_points)
        elif feature_name == 'seeds':
            return (4,)  # Fixed number of seeds
        elif feature_name == 'idx':
            return ()
        else:
            # Default: return None for unknown features
            return None

    def _generate_dummy_array(self, shape, dtype, feature_name):
        """Generate a dummy array with realistic values for the given feature."""
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        if np.issubdtype(dtype, np.complexfloating):
            # Complex features
            real = rng.random(shape).astype(np.float32)
            imag = rng.random(shape).astype(np.float32)
            return (real + 1j * imag).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            # Real-valued features
            if feature_name in ['sourcemap', 'source_strength_analytic', 'source_strength_estimated']:
                # These should be positive
                return rng.exponential(0.1, shape).astype(dtype)
            elif feature_name in ['noise_strength_analytic', 'noise_strength_estimated']:
                # Small positive values
                return rng.exponential(0.01, shape).astype(dtype)
            elif feature_name in ['csmtriu']:
                # Symmetric matrix (upper triangular stored as full)
                data = rng.random(shape).astype(dtype)
                # Make it symmetric
                for i in range(min(shape[-2], shape[-1])):
                    for j in range(i, min(shape[-2], shape[-1])):
                        data[..., i, j] = data[..., j, i]
                return data
            else:
                return rng.random(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            # Integer features
            if feature_name == 'idx':
                return np.array(0, dtype=dtype)
            elif feature_name == 'seeds':
                return rng.randint(0, 1000, shape, dtype=dtype)
            else:
                return rng.randint(0, 100, shape, dtype=dtype)
        else:
            return rng.random(shape).astype(dtype)


class DummyDataCache:
    """Cache for dummy dataset samples.

    This class handles caching of generated dummy data to disk, using a hash
    of the parameters as the cache key.
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, mode, f, num, features, idx):
        """Generate a unique cache key for the given parameters."""
        # Convert parameters to a hashable string
        params = {
            'mode': mode,
            'f': str(f) if f is not None else 'None',
            'num': num,
            'features': tuple(sorted(features)),
            'idx': idx,
        }
        params_str = str(params)
        return hashlib.sha256(params_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key):
        """Get the file path for a cache key."""
        return self.cache_dir / f'{cache_key}.pkl'

    def get(self, mode, f, num, features, idx):
        """Get cached data for the given parameters."""
        cache_key = self._get_cache_key(mode, f, num, features, idx)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                # Cache file is corrupted, remove it
                cache_path.unlink(missing_ok=True)

        return None

    def set(self, mode, f, num, features, idx, data):
        """Cache data for the given parameters."""
        cache_key = self._get_cache_key(mode, f, num, features, idx)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, IOError) as e:
            # Log warning but don't fail
            print(f'Warning: Failed to cache dummy data: {e}')

    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()


class DatasetDummy(DatasetBase):
    """Dummy dataset that returns pre-computed data for fast testing.

    This dataset generates synthetic data with correct shapes and data types,
    caches it to disk, and loads from cache on subsequent runs. This is useful
    for testing TFRecord serialization and other I/O operations without
    triggering the computationally expensive acoustic pipeline.

    Parameters
    ----------
    mode : str
        Type of calculation method ('welch', 'analytic', 'wishart'). Defaults to 'welch'.
    mic_sig_noise : bool
        Apply signal noise to microphone signals. Defaults to True.
    cache_dir : str or Path, optional
        Directory to store cached data. Defaults to ~/.cache/acoupipe/dummy_dataset.
    use_cache : bool
        Whether to use cached data. If False, data is regenerated each time. Defaults to True.
    regenerate_cache : bool
        If True, regenerate and overwrite existing cache. Defaults to False.
    """

    # Additional traits for caching
    cache = Instance(DummyDataCache, desc='Cache for dummy data')
    use_cache = Bool(True, desc='Whether to use cached data')
    regenerate_cache = Bool(False, desc='Whether to regenerate cache')
    _cache_hits = Int(0, desc='Number of cache hits')
    _cache_misses = Int(0, desc='Number of cache misses')

    def __init__(
        self,
        mode='welch',
        mic_sig_noise=True,
        cache_dir=None,
        use_cache=True,
        regenerate_cache=False,
        tasks=1,
        remote_args=None,
        logger=None,
        config=None,
    ):
        if config is None:
            config = DatasetDummyConfig(
                mode=mode,
                mic_sig_noise=mic_sig_noise,
            )
        super().__init__(config=config, tasks=tasks, remote_args=remote_args, logger=logger)

        self.cache = DummyDataCache(cache_dir=cache_dir)
        self.use_cache = use_cache
        self.regenerate_cache = regenerate_cache
        self._cache_hits = 0
        self._cache_misses = 0

    def get_feature_collection(self, features, f, num):
        """Get the feature collection with dummy features."""
        # Handle custom features
        custom_features = [feat for feat in features if isinstance(feat, BaseFeatureCatalog)]
        default_feature_names = [feat for feat in features if isinstance(feat, str)]

        # Get default features from config
        default_features = self.config.get_default_features(default_feature_names, f, num)

        # Build feature collection
        from acoupipe.datasets.features import BaseFeatureCollectionBuilder

        builder = BaseFeatureCollectionBuilder(features=default_features + custom_features)

        # Add prepare function if available
        if hasattr(self.config, 'get_prepare_func'):
            builder.add_custom(self.config.get_prepare_func())

        feature_collection = builder.build()

        # Add cleanup function
        if hasattr(self.config, 'get_cleanup_func'):
            builder.add_custom(self.config.get_cleanup_func(features))

        return feature_collection

    def generate(self, features, size, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Generate dummy dataset samples.

        This method either loads data from cache or generates new dummy data
        with correct shapes and data types.

        Parameters
        ----------
        features : list
            List of features included in the dataset.
        size : int
            Size of the dataset (number of samples).
        split : str
            Split name for the dataset ('training', 'validation' or 'test').
        f : float or list or None
            The center frequency or list of frequencies.
        num : int
            Controls the width of the frequency bands.
        start_idx : int
            Starting sample index.
        progress_bar : bool
            Whether to show a progress bar.

        Yields
        ------
        dict
            Dataset samples as dictionaries containing the feature names as keys.
        """
        # Add mandatory features
        all_features = set(features)
        all_features.add('idx')
        all_features.add('seeds')
        all_features = list(all_features)

        for idx in range(start_idx, start_idx + size):
            # Try to get from cache
            cached_data = None
            if self.use_cache:
                if not self.regenerate_cache:
                    cached_data = self.cache.get(self.config.mode, f, num, all_features, idx)

                if cached_data is not None:
                    self._cache_hits += 1
                    data = cached_data.copy()
                else:
                    self._cache_misses += 1
                    # Generate new dummy data
                    data = self._generate_dummy_sample(idx, all_features, f, num)

                    # Cache it for future use (even when regenerating)
                    self.cache.set(self.config.mode, f, num, all_features, idx, data)
            else:
                # Generate new dummy data without caching
                data = self._generate_dummy_sample(idx, all_features, f, num)

            # Filter to requested features only
            yield {k: v for k, v in data.items() if k in features or k in ['idx', 'seeds']}

    def _generate_dummy_sample(self, idx, features, f, num):
        """Generate a single dummy sample with the requested features."""
        sample = {}

        # Generate each requested feature
        for feature in features:
            sample[feature] = self._generate_dummy_feature(feature, f, num, idx)

        # Ensure idx and seeds are always present
        if 'idx' not in sample:
            sample['idx'] = np.array(idx, dtype=np.int64)
        if 'seeds' not in sample:
            sample['seeds'] = np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.int64)

        return sample

    def _generate_dummy_feature(self, feature_name, f, num, idx):
        """Generate a dummy feature array with correct shape and dtype."""
        shape = self.config._get_feature_shape(feature_name, f, num)
        dtype = self.config.feature_dtypes[feature_name]

        # Use a deterministic seed based on feature name and idx for reproducibility
        seed = hash(f'{feature_name}_{idx}_{f}_{num}') % (2**32)
        rng = np.random.RandomState(seed)

        if feature_name == 'f':
            # Frequency array
            if f is None:
                f_values = np.linspace(100, 10000, self.config.nfft, dtype=np.float32)
            elif isinstance(f, list):
                f_values = np.array(f, dtype=np.float32)
            else:
                f_values = np.array([f], dtype=np.float32)
            return f_values

        elif feature_name == 'num':
            return np.array(num, dtype=np.int64)

        elif feature_name == 'idx':
            return np.array(idx, dtype=np.int64)

        elif feature_name == 'seeds':
            return np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.int64)

        elif np.issubdtype(dtype, np.complexfloating):
            # Complex features
            shape = tuple(s if s is not None else 1 for s in shape)
            real = rng.random(shape).astype(np.float32)
            imag = rng.random(shape).astype(np.float32)
            return (real + 1j * imag).astype(dtype)

        elif np.issubdtype(dtype, np.floating):
            # Real-valued features
            shape = tuple(s if s is not None else 1 for s in shape)
            if feature_name in ['sourcemap', 'source_strength_analytic', 'source_strength_estimated']:
                # These should be positive
                return rng.exponential(0.1, shape).astype(dtype)
            elif feature_name in ['noise_strength_analytic', 'noise_strength_estimated']:
                # Small positive values
                return rng.exponential(0.01, shape).astype(dtype)
            elif feature_name in ['csmtriu']:
                # Symmetric matrix (upper triangular stored as full)
                data = rng.random(shape).astype(dtype)
                # Make it symmetric
                for i in range(min(shape[-2], shape[-1])):
                    for j in range(i, min(shape[-2], shape[-1])):
                        data[..., i, j] = data[..., j, i]
                return data
            else:
                return rng.random(shape).astype(dtype)

        elif np.issubdtype(dtype, np.integer):
            # Integer features
            shape = tuple(s if s is not None else 1 for s in shape)
            return rng.randint(0, 100, shape, dtype=dtype)

        else:
            shape = tuple(s if s is not None else 1 for s in shape)
            return rng.random(shape).astype(dtype)

    def get_cache_stats(self):
        """Get cache hit/miss statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0,
        }

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def save_tfrecord(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a .tfrecord file using dummy data.

        This overrides the base class method to use the dummy dataset's generate method
        instead of creating a pipeline, since the dummy dataset doesn't use real acoustic calculations.
        """
        from acoupipe.config import TF_FLAG

        if not TF_FLAG:
            msg = 'TensorFlow is not available. Cannot save to TFRecord format.'
            raise ImportError(msg)

        import tensorflow as tf
        from acoupipe.writer import WriteTFRecord, complex_list_feature

        # Get the feature collection to get the TF shape and dtype mappings
        feature_collection = self.get_feature_collection(features, f, num)

        # Get features with varying length to handle them correctly in the TFRecord writer
        shape_features = []
        for feature, shape in feature_collection.feature_tf_shape_mapper.items():
            # if more than one None in shape, we have a varying length feature
            if list(shape).count(None) > 1:
                shape_features.append(feature)

        # Create a dummy pipeline that uses our generate method
        from acoupipe.base import DataGenerator
        from traits.api import Any, Bool, Int, List, Str

        class DummyPipeline(DataGenerator):
            # Define traits for the attributes we need
            dummy_dataset = Any(desc='Reference to the dummy dataset')
            features = List(Str, desc='List of features to generate')
            size = Int(desc='Number of samples to generate')
            split = Str(desc='Dataset split')
            f = Any(desc='Frequency parameter')
            num = Int(desc='Num parameter')
            start_idx = Int(desc='Starting index')

            def __init__(self, dummy_dataset, features, size, split, f, num, start_idx):
                self.dummy_dataset = dummy_dataset
                self.features = features
                self.size = size
                self.split = split
                self.f = f
                self.num = num
                self.start_idx = start_idx

            def get_data(self, progress_bar=True, start_idx=None):
                if start_idx is None:
                    start_idx = self.start_idx
                yield from self.dummy_dataset.generate(
                    features=self.features,
                    size=self.size,
                    split=self.split,
                    f=self.f,
                    num=self.num,
                    start_idx=start_idx,
                    progress_bar=progress_bar,
                )

        # Create the dummy pipeline
        pipeline = DummyPipeline(self, features, size, split, f, num, start_idx)

        # Use the WriteTFRecord with our dummy pipeline
        WriteTFRecord(
            name=name,
            source=pipeline,
            shape_features=shape_features,
            encoder_funcs=feature_collection.feature_tf_encoder_mapper,
        ).save(progress_bar, start_idx)

    def save_h5(self, features, size, name, split='training', f=None, num=0, start_idx=0, progress_bar=True):
        """Save dataset to a HDF5 file using dummy data.

        This overrides the base class method to use the dummy dataset's generate method
        instead of creating a pipeline, since the dummy dataset doesn't use real acoustic calculations.
        """
        from acoupipe.base import DataGenerator
        from acoupipe.writer import WriteH5Dataset
        from traits.api import Any, Int, List, Str

        # Create a dummy pipeline that uses our generate method
        class DummyPipeline(DataGenerator):
            # Define traits for the attributes we need
            dummy_dataset = Any(desc='Reference to the dummy dataset')
            features = List(Str, desc='List of features to generate')
            size = Int(desc='Number of samples to generate')
            split = Str(desc='Dataset split')
            f = Any(desc='Frequency parameter')
            num = Int(desc='Num parameter')
            start_idx = Int(desc='Starting index')

            def __init__(self, dummy_dataset, features, size, split, f, num, start_idx):
                self.dummy_dataset = dummy_dataset
                self.features = features
                self.size = size
                self.split = split
                self.f = f
                self.num = num
                self.start_idx = start_idx

            def get_data(self, progress_bar=True, start_idx=None):
                if start_idx is None:
                    start_idx = self.start_idx
                yield from self.dummy_dataset.generate(
                    features=self.features,
                    size=self.size,
                    split=self.split,
                    f=self.f,
                    num=self.num,
                    start_idx=start_idx,
                    progress_bar=progress_bar,
                )

        # Create the dummy pipeline
        pipeline = DummyPipeline(self, features, size, split, f, num, start_idx)

        # Use the WriteH5Dataset with our dummy pipeline
        WriteH5Dataset(
            name=name,
            source=pipeline,
        ).save(progress_bar, start_idx)

    def get_tf_dataset(self, features, size, split='training', f=None, num=0, start_idx=0, progress_bar=False):
        """Get a TensorFlow dataset from dummy data.

        This overrides the base class method to use the dummy dataset's generate method
        instead of creating a pipeline, since the dummy dataset doesn't use real acoustic calculations.
        """
        from acoupipe.config import TF_FLAG

        if not TF_FLAG:
            msg = 'TensorFlow is not available. Cannot create TF dataset.'
            raise ImportError(msg)

        import tensorflow as tf
        from functools import partial

        # Get the feature collection to get the output signature
        feature_collection = self.get_feature_collection(features, f, num)

        # Add mandatory features
        all_features = list(features) + ['idx', 'seeds']

        # Get output signature
        output_signature = self.get_output_signature(all_features, f=f, num=num)

        # Create a generator function that uses our dummy generate method
        def _generate_dummy():
            for data in self.generate(
                features=all_features,
                size=size,
                split=split,
                f=f,
                num=num,
                start_idx=start_idx,
                progress_bar=progress_bar,
            ):
                yield data

        return tf.data.Dataset.from_generator(
            _generate_dummy,
            output_signature=output_signature,
        )
