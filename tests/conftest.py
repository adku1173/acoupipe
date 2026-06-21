import os
import shutil
import tempfile
from pathlib import Path

from acoular import ImportGrid, SteeringVector
from acoupipe.datasets.experimental import DatasetMIRACLE
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.synthetic import DatasetSynthetic
from acoupipe.sampler import ContainerSampler, LocationSampler, NumericAttributeSampler
from acoupipe.writer import WriteH5Dataset

from tests.miracle_test_config import DatasetMIRACLETestConfig
from tests.synthetic_test_config import DatasetSyntheticTestConfig

from .constants import MIC_GEOM
from .dummy_dataset import DatasetDummy
from .pipeline_value_test import get_pipeline

import numpy as np
import pytest
from scipy.stats import norm


class _AttributeTarget:
    """Simple target used in sampler fixtures."""

    attribute = 0


def _create_container_method(target_instance):
    def sample_method(random_state):
        target_instance.attribute = random_state.random()

    return sample_method


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


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
def create_dummy_dataset():
    """Create a DatasetDummy instance for fast testing without expensive calculations."""

    def _create_dataset(mode='welch', mic_sig_noise=True, use_cache=True, cache_dir=None, **kwargs):
        return DatasetDummy(mode=mode, mic_sig_noise=mic_sig_noise, use_cache=use_cache, cache_dir=cache_dir, **kwargs)

    return _create_dataset


@pytest.fixture
def create_miracle_dataset():
    """Create a DatasetMIRACLE instance for tests using the smaller D1 scenario by default."""

    def _create_dataset(full=True, tasks=1, **kwargs):
        kwargs.setdefault('scenario', 'D1')
        if full:
            return DatasetMIRACLE(tasks=tasks, **kwargs)
        config = DatasetMIRACLETestConfig(**kwargs)
        return DatasetMIRACLE(config=config, tasks=tasks, **kwargs)

    return _create_dataset


@pytest.fixture
def psa():
    psa_instance = PowerSpectraAnalytic(
        block_size=256,
        overlap='50%',
        sample_freq=51200,
        num_samples=51200,
        steer=SteeringVector(mics=MIC_GEOM, grid=ImportGrid(pos=np.random.normal(size=(3, 3)))),
    )
    nfft = psa_instance.fftfreq()
    psa_instance.Q = np.stack([np.eye(3, dtype=complex) * 0.1 for _ in range(nfft.shape[0])])
    psa_instance.noise = np.stack([np.eye(MIC_GEOM.num_mics, dtype=complex) for _ in range(nfft.shape[0])])
    return psa_instance


@pytest.fixture
def location_sampler():
    return LocationSampler(
        random_var=(norm(0, 0.1688), norm(0, 0.1688), norm(0.5, 0)),
        x_bounds=(-0.5, 0.5),
        y_bounds=(-0.5, 0.5),
        z_bounds=(0.5, 0.5),
        nsources=3,
        random_state=np.random.RandomState(1),
    )


@pytest.fixture
def numeric_attribute_sampler():
    sampler = NumericAttributeSampler(random_var=norm(loc=0, scale=0.1688), random_state=5, attribute='attribute')
    sampler.target = [_AttributeTarget() for _ in range(10)]
    return sampler


@pytest.fixture
def container_sampler():
    target = _AttributeTarget()
    sampler = ContainerSampler()
    sampler.random_func = _create_container_method(target)
    return sampler, target


@pytest.fixture
def base_pipeline():
    size = 1
    pipeline = get_pipeline(size)
    test_seeds = {1: range(1, 1 + size), 2: range(2, 2 + size), 3: range(3, 3 + size), 4: range(4, 4 + size)}
    return pipeline, test_seeds


@pytest.fixture
def h5_test_file():
    """Create and clean up a temporary HDF5 test file."""
    pipeline = get_pipeline(5)
    pipeline.random_seeds = {
        1: range(1, 6),
        2: range(2, 7),
        3: range(3, 8),
        4: range(4, 9),
    }
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, 'test_data.h5')
    writer = WriteH5Dataset(source=pipeline, name=file_path)
    writer.save()
    try:
        yield file_path
    finally:
        os.remove(file_path)
        os.rmdir(temp_dir)
