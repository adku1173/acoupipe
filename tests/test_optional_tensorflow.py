"""TensorFlow must only be imported when TFRecord files or TensorFlow datasets are used."""

import subprocess
import sys

MODULES = [
    'acoupipe.datasets',
    'acoupipe.datasets.ism.dataset',
    'acoupipe.datasets.miracle.dataset',
    'acoupipe.datasets.sriracha.dataset',
    'acoupipe.datasets.synthetic.dataset',
    'acoupipe.loader',
    'acoupipe.pipeline',
    'acoupipe.sampler',
    'acoupipe.writer',
]


def test_importing_acoupipe_does_not_import_tensorflow():
    """Ray workers import AcouPipe to deserialize feature functions and must not load TensorFlow."""
    imports = '; '.join(f'import {module}' for module in MODULES)
    code = f"{imports}; import sys; print('tensorflow' in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)  # noqa: S603
    assert result.stdout.strip() == 'False'
