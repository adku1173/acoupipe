"""Seeding of samplers by dataset seed, split, sample index and sampler key."""

from acoupipe.datasets import Config, Dataset
from acoupipe.datasets.base.parameters import ParameterSet
from acoupipe.datasets.synthetic import DatasetSyntheticConfig

import pytest
from scipy.stats import norm


def create_config(paths):
    """Create a Config that samples the given parameter paths from a standard normal."""
    config = Config(parameters=ParameterSet(a=0.0, b=0.0))
    for path in paths:
        config.sample(path, random_var=norm())
    return config


def generate(paths, size=4, **kwargs):
    """Generate the sampled values of a and b, keyed by sample index."""
    seed = kwargs.pop('seed', 0)
    dataset = Dataset(config=create_config(paths), seed=seed)
    samples = dataset.generate(features=['a', 'b'], size=size, progress_bar=False, **kwargs)
    return {sample['idx']: (float(sample['a']), float(sample['b'])) for sample in samples}


def test_samples_of_a_sampler_do_not_depend_on_other_samplers():
    """Removing or reordering samplers leaves the samples of the remaining samplers unchanged."""
    both = generate(['a', 'b'])
    reordered = generate(['b', 'a'])
    only_b = generate(['b'])
    assert reordered == both
    assert [b for _, b in only_b.values()] == [b for _, b in both.values()]


def test_samples_are_reproducible_and_independent_of_start_index():
    """A sample is determined by seed, split and index, independently of the generated range."""
    full = generate(['a', 'b'], size=5)
    assert generate(['a', 'b'], size=5) == full
    assert generate(['a', 'b'], size=3, start_idx=2) == {idx: full[idx] for idx in (2, 3, 4)}


@pytest.mark.parametrize('kwargs', [{'split': 'test'}, {'split': 'validation'}, {'seed': 1}])
def test_splits_and_seeds_yield_different_samples(kwargs):
    """Different split names and dataset seeds yield different random streams."""
    training = generate(['a', 'b'])
    other = generate(['a', 'b'], **kwargs)
    assert all(other[idx] != training[idx] for idx in training)


def test_parameter_path_can_be_sampled_once():
    """A parameter path identifies the random stream of its sampler and must therefore be unique."""
    with pytest.raises(ValueError, match='already sampled'):
        create_config(['a', 'a'])


def test_legacy_config_samples_in_key_order():
    """Legacy configs keep their fixed sampler keys, which also define the sampling order."""
    sampler = DatasetSyntheticConfig(mic_pos_noise=True, min_nsources=1, max_nsources=4).get_sampler()
    assert list(sampler) == sorted(sampler)
