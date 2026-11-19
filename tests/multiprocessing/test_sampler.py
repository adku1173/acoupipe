from acoupipe.pipeline import DistributedPipeline

import pytest

pytestmark = pytest.mark.multiprocessing


def test_distributed_pipeline_with_default_seed(distributed_pipeline):
    data = next(distributed_pipeline.get_data(progress_bar=False))
    assert data['data']


@pytest.mark.parametrize(
    'finput',
    [
        lambda sampler: {'res': True},
        (lambda sampler, x: {'res': x}, True),
    ],
)
def test_distributed_pipeline_valid_pipeline_funcs(finput):
    pipeline = DistributedPipeline(numsamples=2, feature_func=finput)
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
def test_distributed_pipeline_invalid_pipeline_funcs(finput):
    pipeline = DistributedPipeline(numsamples=2, feature_func=finput)
    with pytest.raises(ValueError):
        next(pipeline.get_data(progress_bar=False))
