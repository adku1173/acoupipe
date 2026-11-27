"""Base dataset package.

This package owns the shared dataset base class and configuration classes.
"""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.base.dataset import Dataset
from acoupipe.datasets.base.legacy_config import ConfigBase
from acoupipe.datasets.base.legacy_dataset import DatasetBase
from acoupipe.datasets.base.parameters import ParameterSet

__all__ = ['Config', 'ConfigBase', 'Dataset', 'DatasetBase', 'ParameterSet']
