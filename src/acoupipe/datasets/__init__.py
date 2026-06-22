"""The datasets subpackage contains default classes to create acoustical datasets with microphone array data."""

from acoupipe.datasets.base import Config, ConfigBase, DatasetBase, ParameterSet
from acoupipe.datasets.ism.config import SyntheticISMConfig
from acoupipe.datasets.ism.parameters import SyntheticISMParameters
from acoupipe.datasets.miracle.config import MIRACLEConfig
from acoupipe.datasets.miracle.parameters import MIRACLEParameters
from acoupipe.datasets.sriracha.config import SRIRACHAConfig
from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters
from acoupipe.datasets.synthetic.config import SyntheticConfig
from acoupipe.datasets.synthetic.parameters import SyntheticParameters

Dataset = DatasetBase

__all__ = [
    'Config',
    'ConfigBase',
    'Dataset',
    'DatasetBase',
    'DatasetMIRACLE',
    'DatasetSRIRACHA',
    'DatasetSynthetic',
    'DatasetSyntheticConfig',
    'DatasetSyntheticISM',
    'DatasetSyntheticISMConfig',
    'MIRACLEConfig',
    'MIRACLEParameters',
    'ParameterSet',
    'SRIRACHAConfig',
    'SRIRACHAParameters',
    'SyntheticConfig',
    'SyntheticISMConfig',
    'SyntheticISMParameters',
    'SyntheticParameters',
]


def __getattr__(name):
    if name == 'DatasetMIRACLE':
        from acoupipe.datasets.miracle.dataset import DatasetMIRACLE  # noqa: PLC0415

        return DatasetMIRACLE
    if name == 'DatasetSRIRACHA':
        from acoupipe.datasets.sriracha.dataset import DatasetSRIRACHA  # noqa: PLC0415

        return DatasetSRIRACHA
    if name == 'DatasetSynthetic':
        from acoupipe.datasets.synthetic.dataset import DatasetSynthetic  # noqa: PLC0415

        return DatasetSynthetic
    if name == 'DatasetSyntheticConfig':
        from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig  # noqa: PLC0415

        return DatasetSyntheticConfig
    if name == 'DatasetSyntheticISM':
        from acoupipe.datasets.ism.dataset import DatasetSyntheticISM  # noqa: PLC0415

        return DatasetSyntheticISM
    if name == 'DatasetSyntheticISMConfig':
        from acoupipe.datasets.ism.legacy_config import DatasetSyntheticISMConfig  # noqa: PLC0415

        return DatasetSyntheticISMConfig
    raise AttributeError(name)
