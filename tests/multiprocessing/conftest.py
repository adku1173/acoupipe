import sys

from tests.multiprocessing.pipeline_value_test import get_distributed_pipeline

import pytest


@pytest.fixture
def distributed_pipeline():
    # Skip Ray-dependent tests on macOS due to Ray initialization issues.
    if sys.platform == 'darwin':
        pytest.skip('Ray tests are not stable on macOS runners')
    return get_distributed_pipeline(3, 2)
