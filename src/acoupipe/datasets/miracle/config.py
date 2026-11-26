"""MIRACLE dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.miracle.parameters import MIRACLEParameters

from traits.api import Int


class MIRACLEConfig(Config):
    """High-level Config using MIRACLE default parameters."""

    # Temporary: remove together with Config._get_fftfreq once the spectral features are migrated.
    _fft_block_size = Int(1024, desc='temporary FFT block size of the legacy defaults')

    def create_parameters(self):
        """Create MIRACLE parameters for this config."""
        return MIRACLEParameters()


__all__ = ['MIRACLEConfig']
