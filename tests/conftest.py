import os
import shutil
import sys
import tempfile
from pathlib import Path

from acoular import ImportGrid, MicGeom, SteeringVector
import numpy as np
import pytest
from scipy.stats import norm

from acoupipe.datasets.experimental import DatasetMIRACLE
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.synthetic import DatasetSynthetic, DatasetSyntheticTestConfig
from acoupipe.sampler import ContainerSampler, LocationSampler, NumericAttributeSampler
from acoupipe.writer import WriteH5Dataset

from .pipeline_value_test import get_distributed_pipeline, get_pipeline

MIC_GEOM = MicGeom(
    pos_total=np.array(
        [
            [-0.68526741, -0.7593943, -1.99918406, 0.08414458],
            [-0.60619132, 1.20374544, -0.27378946, -1.38583541],
            [0.32909911, 0.56201909, -0.24697204, -0.68677001],
        ]
    )
)


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
def create_miracle_dataset():
    """Create a DatasetMIRACLE instance for tests."""

    def _create_dataset(tasks=1, **kwargs):
        return DatasetMIRACLE(tasks=tasks, **kwargs)

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
def distributed_pipeline():
    # Skip Ray-dependent tests on macOS due to Ray initialization issues
    if sys.platform == 'darwin':
        pytest.skip('Ray tests are not stable on macOS runners')
    size = 3
    pipeline = get_distributed_pipeline(size, 2)  # two workers
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
