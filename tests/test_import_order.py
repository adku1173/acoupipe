"""Acoular must configure OpenBLAS before AcouPipe loads NumPy."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', ['acoupipe.datasets', 'acoupipe.pipeline'])
def test_acoular_configures_openblas_on_first_import(module):
    """Import each entry point fresh, without pytest's NumPy imports or thread settings."""
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '2'}
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            '-W',
            'error:We detected that Numpy is already loaded:UserWarning',
            '-c',
            f'import {module}; import os; print(os.environ["OPENBLAS_NUM_THREADS"])',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1'
