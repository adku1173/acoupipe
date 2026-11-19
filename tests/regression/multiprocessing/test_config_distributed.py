"""Distributed generation must see the same sampled ParameterSet values as sequential generation."""

from acoupipe.datasets import Config, Dataset
from acoupipe.datasets.base.parameters import ParameterSet

import numpy as np
import pytest

pytestmark = pytest.mark.multiprocessing

FEATURES = ['c', 'c_feature', 'wavelength']


def create_config():
    """Create a Config whose sampled value is read by a feature and a prepare callback."""
    config = Config(parameters=ParameterSet(c=343.0))
    config.sample('c', random_func=lambda rng: float(rng.uniform(300.0, 400.0)))

    @config.prepare
    def wavelength(parameters):
        return {'wavelength': parameters.c / 1000.0}

    config.feature('c_feature', lambda parameters: parameters.c)
    config.feature('wavelength', lambda data: data['wavelength'])
    return config


def generate(tasks, size=4, samples_per_task=16):
    """Generate samples with the given number of tasks, keyed by sample index."""
    dataset = Dataset(config=create_config(), tasks=tasks, samples_per_task=samples_per_task)
    samples = dataset.generate(features=FEATURES, size=size, progress_bar=False)
    return {sample['idx']: sample for sample in samples}


def test_distributed_samples_reach_parameter_outputs_features_and_prepare_callbacks():
    """Each actor must sample into the ParameterSet that its features and prepare callbacks read."""
    sequential = generate(tasks=1)
    distributed = generate(tasks=2)
    assert distributed.keys() == sequential.keys()
    for idx, expected in sequential.items():
        for name in FEATURES:
            assert float(distributed[idx][name]) == float(expected[name])
    assert len({float(sample['c']) for sample in distributed.values()}) == len(distributed)


@pytest.mark.parametrize('samples_per_task', [1, 3])
def test_samples_per_task_matches_sequential_generation(samples_per_task):
    """Tasks with several samples, including shorter final tasks, reproduce sequential samples."""
    sequential = generate(tasks=1, size=7)
    distributed = generate(tasks=2, size=7, samples_per_task=samples_per_task)
    assert distributed.keys() == sequential.keys()
    for idx, expected in sequential.items():
        for name in FEATURES:
            assert np.array_equal(distributed[idx][name], expected[name])
