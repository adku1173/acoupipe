from acoular import MicGeom, PointSource, SourceMixer, WNoiseGenerator
from acoupipe.pipeline import BasePipeline
from acoupipe.sampler import (
    ContainerSampler,
    MicGeomSampler,
    NumericAttributeSampler,
    PointSourceSampler,
    SetSampler,
    SourceSetSampler,
)

import numpy as np
import pytest
from numpy.random import RandomState, default_rng
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import cdist
from scipy.stats import norm

SAMPLER_CLASSES = [
    NumericAttributeSampler,
    SetSampler,
    SourceSetSampler,
    PointSourceSampler,
    ContainerSampler,
    MicGeomSampler,
]
STATES = [1, RandomState(1), default_rng(1)]

FLOAT_SET = [0.1, 0.2, 0.3, 0.4]  # a set with floats for testing


class LinkedTarget:
    """Class that is used in Sampler tests."""

    linked_attribute = 0


class Target:
    """Class that is used in Sampler tests."""

    attribute = 0


def test_location_sampler_mindist(location_sampler):
    """Test if minimum distance is respected."""
    location_sampler.sample()
    dist = cdist(location_sampler.target.T, location_sampler.target.T).max()
    assert np.min(dist) < 0.4

    location_sampler.mindist = 0.4
    location_sampler.sample()
    dist = cdist(location_sampler.target.T, location_sampler.target.T).max()
    assert np.min(dist) > 0.4


@pytest.mark.parametrize('equal_value', [False, True])
def test_set_sampler_equal_value(equal_value):
    """Test equal_value capabilities."""
    sampler = SetSampler(random_state=RandomState(1), set=FLOAT_SET, attribute='attribute', replace=False)
    sampler.target = [Target() for _ in range(3)]
    sampler.equal_value = equal_value
    sampler.sample()
    samples = [tr.attribute for tr in sampler.target]
    if equal_value:
        assert len(set(samples)) == 1
    else:
        assert len(set(samples)) > 1


def test_set_sampler_sampling_linked_attributes():
    """Test assignment to linked attributes."""
    linked_target = LinkedTarget()
    target = Target()
    target.attribute = linked_target
    sampler = SetSampler(
        random_state=RandomState(1), set=FLOAT_SET, attribute='attribute.linked_attribute', replace=False
    )
    sampler.target = [target]
    sampler.sample()
    assert linked_target.linked_attribute != 0


@pytest.mark.parametrize(
    'equal_value, nsources, expected_setsize',
    [
        (False, 1, 1),
        (True, 1, 1),
        (False, 2, 2),
        (False, 2, 2),
    ],
)
def test_source_set_sampler(equal_value, nsources, expected_setsize):
    ps_set = [PointSource(signal=WNoiseGenerator(sample_freq=10), mics=MicGeom()) for _ in range(4)]
    sampler = SourceSetSampler(
        target=[SourceMixer(), SourceMixer()],
        set=ps_set,
        replace=False,
        random_state=RandomState(1),
        nsources=nsources,
        equal_value=equal_value,
    )
    sampler.sample()
    l1 = len(sampler.target[0].sources)
    l2 = len(sampler.target[1].sources)
    assert (l1, l2) == (expected_setsize, expected_setsize)
    if equal_value:  # assert that same sources at every target
        assert sampler.target[0].sources == sampler.target[1].sources
    else:
        assert sampler.target[0].sources != sampler.target[1].sources


def test_numeric_attribute_sampler_order(numeric_attribute_sampler):
    """Assert that ordering of numeric samples works."""
    sampler = numeric_attribute_sampler
    sampler.sample()
    l = [t.attribute for t in sampler.target]
    assert l[0] != min(l) and l[-1] != max(l)

    sampler.order = 'ascending'
    sampler.sample()
    l = [t.attribute for t in sampler.target]
    assert l[0] == min(l) and l[-1] == max(l)

    sampler.order = 'descending'
    sampler.sample()
    l = [t.attribute for t in sampler.target]
    assert l[0] == max(l) and l[-1] == min(l)


def test_numeric_attribute_sampler_normalization(numeric_attribute_sampler):
    """Verifies that normalization works."""
    sampler = numeric_attribute_sampler
    sampler.normalize = True
    sampler.sample()
    l = [t.attribute for t in sampler.target]
    assert max(l) == 1.0


@pytest.mark.parametrize(
    'rng1, rng2',
    [
        (RandomState(10), RandomState(10)),
        (default_rng(5), default_rng(5)),
    ],
)
def test_container_sampler_sampling(container_sampler, rng1, rng2):
    sampler, target = container_sampler
    sampler.random_state = rng1
    sampler.sample()
    assert target.attribute == rng2.random()


def test_container_sampler_error_handling(container_sampler):
    """Tests if ValueError is thrown for wrong random_func input."""
    sampler, _ = container_sampler

    def random_func(rng, x):
        pass

    sampler.random_func = random_func
    with pytest.raises(ValueError):
        sampler.sample()


@pytest.mark.parametrize('mode', ['deviate', 'rotate', 'translate'])
def test_micgeom_sampler(mode):
    rng = RandomState(1)
    micgeom = MicGeom(pos_total=rng.rand(3, 10))
    digest1 = micgeom.digest

    rng = RandomState(2)
    normal_distribution = norm(loc=0, scale=0.1)
    if mode == 'deviate':
        sampler = MicGeomSampler(random_var=normal_distribution, random_state=rng, ddir=np.array([[1.0], [1.0], [1.0]]))
    elif mode == 'rotate':
        sampler = MicGeomSampler(random_var=normal_distribution, random_state=rng, rvec=np.array([[1.0], [1.0], [1.0]]))
    elif mode == 'translate':
        sampler = MicGeomSampler(random_var=normal_distribution, random_state=rng, tdir=np.array([[1.0], [1.0], [1.0]]))

    sampler.target = micgeom
    mpos_target = micgeom.pos_total.copy()
    sampler.sample()
    digest2 = micgeom.digest
    assert_almost_equal(mpos_target, sampler.mpos_init)
    assert micgeom.pos_total[0, 0] != mpos_target[0, 0]
    assert digest1 != digest2


@pytest.mark.parametrize('cls', SAMPLER_CLASSES)
def test_sampler_instancing(cls):
    """Create an instance of each class defined in module."""
    cls()


@pytest.mark.parametrize('cls', SAMPLER_CLASSES)
@pytest.mark.parametrize('state', STATES)
def test_sampler_seeding(cls, state):
    """Tests if different random states can be assigned to sampler objects."""
    if cls not in [SetSampler, SourceSetSampler, ContainerSampler]:
        cls(random_state=state)


def test_pipeline_without_explicit_seeds(base_pipeline):
    pipeline, _ = base_pipeline
    data = next(pipeline.get_data(progress_bar=False))
    assert data['data']


def test_too_short_random_seeds_input(base_pipeline):
    pipeline, _ = base_pipeline
    seeds = {1: range(1, 2)}
    pipeline.random_seeds = seeds
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))


def test_non_equal_length_random_seeds_input(base_pipeline):
    pipeline, test_seeds = base_pipeline
    test_seeds[0] = range(0, 10)
    pipeline.random_seeds = test_seeds
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))


@pytest.mark.parametrize(
    'finput',
    [
        lambda sampler: {'res': True},
        (lambda sampler, x: {'res': x}, True),
    ],
)
def test_valid_pipeline_funcs(base_pipeline, finput):
    _, _ = base_pipeline
    pipeline = BasePipeline(numsamples=2, features=finput)
    data = next(pipeline.get_data(progress_bar=False))
    assert data['res']


@pytest.mark.parametrize(
    'finput',
    [
        None,
        lambda: {'res': True},
        (lambda sampler, x: {'res': x}, True, True),
    ],
)
def test_invalid_pipeline_funcs(base_pipeline, finput):
    _, _ = base_pipeline
    pipeline = BasePipeline(numsamples=2, features=finput)
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))
