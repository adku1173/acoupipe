import sys

import pytest

from tests.multiprocessing.pipeline_value_test import get_distributed_pipeline


@pytest.fixture
def distributed_pipeline():
    # Skip Ray-dependent tests on macOS due to Ray initialization issues.
    if sys.platform == 'darwin':
        pytest.skip('Ray tests are not stable on macOS runners')
    size = 3
    pipeline = get_distributed_pipeline(size, 2)
    test_seeds = {1: range(1, 1 + size), 2: range(2, 2 + size), 3: range(3, 3 + size), 4: range(4, 4 + size)}
    return pipeline, test_seeds
