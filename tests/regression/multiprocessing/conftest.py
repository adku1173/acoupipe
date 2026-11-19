"""Run the multiprocessing regressions in a dedicated two-CPU Ray cluster."""

import sys

import pytest
import ray


@pytest.fixture(scope='package', autouse=True)
def ray_cluster():
    """Reuse a small local Ray cluster for the multiprocessing regressions."""
    with pytest.MonkeyPatch.context() as env:
        if sys.platform == 'darwin':
            # On macOS CI, raylet times out waiting for the metrics export port.
            env.setenv('RAY_enable_metrics_collection', '0')
        ray.init(address='local', num_cpus=2, include_dashboard=False, object_store_memory=256 * 1024**2)
        try:
            yield
        finally:
            ray.shutdown()
