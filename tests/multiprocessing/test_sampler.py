import pytest

from acoupipe.pipeline import DistributedPipeline


pytestmark = pytest.mark.multiprocessing


def test_distributed_pipeline_without_explicit_seeds(distributed_pipeline):
    pipeline, _ = distributed_pipeline
    data = next(pipeline.get_data(progress_bar=False))
    assert data['data']


def test_distributed_pipeline_too_short_random_seeds_input(distributed_pipeline):
    pipeline, _ = distributed_pipeline
    seeds = {1: range(1, 2)}
    pipeline.random_seeds = seeds
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))


def test_distributed_pipeline_non_equal_length_random_seeds_input(distributed_pipeline):
    pipeline, test_seeds = distributed_pipeline
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
def test_distributed_pipeline_valid_pipeline_funcs(distributed_pipeline, finput):
    _, _ = distributed_pipeline
    pipeline = DistributedPipeline(numsamples=2, features=finput)
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
def test_distributed_pipeline_invalid_pipeline_funcs(distributed_pipeline, finput):
    _, _ = distributed_pipeline
    pipeline = DistributedPipeline(numsamples=2, features=finput)
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))
