"""SRIRACHA dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters

from traits.api import Int


class SRIRACHAConfig(Config):
    """High-level Config using SRIRACHA default parameters."""

    # Temporary: remove together with Config._get_fftfreq once the spectral features are migrated.
    _fft_block_size = Int(1024, desc='temporary FFT block size of the legacy defaults')

    def create_parameters(self):
        """Create SRIRACHA parameters for this config."""
        return SRIRACHAParameters()


__all__ = ['SRIRACHAConfig']
