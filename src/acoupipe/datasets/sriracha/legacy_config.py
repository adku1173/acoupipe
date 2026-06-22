"""Legacy SRIRACHA dataset configuration."""

from acoupipe.datasets.miracle.legacy_config import DatasetMIRACLEConfig

from irdl import SrirachaDataset
from traits.api import Either

_SRIRACHA_SCENARIOS = [
    'SR1',
    'SR1-C1',
    'SR1-C2',
    'SR1-C3',
    'SR1-C4',
    'SR1-D',
    'SR2',
    'SR2-C1',
    'SR2-C2',
    'SR2-C3',
    'SR2-C4',
    'SR2-D',
    'SRA1',
    'SRA1-C1',
    'SRA1-C2',
    'SRA1-C3',
    'SRA1-C4',
    'SRA1-D',
    'SRA2',
    'SRA2-C1',
    'SRA2-C2',
    'SRA2-C3',
    'SRA2-C4',
    'SRA2-D',
]


class DatasetSRIRACHAConfig(DatasetMIRACLEConfig):
    """Configuration class for the DatasetSRIRACHA dataset."""

    scenario = Either(_SRIRACHA_SCENARIOS, default='SR1-D', desc='experimental configuration')
    dataset_split = Either(None, 'C1', 'C2', 'C3', 'C4', default=None, desc='artificial dataset split')

    def set_filename(self):
        """Resolve the SRIR file path, downloading via :mod:`irdl` if necessary."""
        output_format = 'raw' if self.dataset_split is not None or self.scenario.endswith('D') else 'hdf5'

        self._filename = str(
            SrirachaDataset.get(
                scenario=self.scenario,
                dataset_split=self.dataset_split,
                cache_dir=self.srir_dir,
                output_format=output_format,
            )
        )


__all__ = ['DatasetSRIRACHAConfig']
