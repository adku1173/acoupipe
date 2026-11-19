"""SRIRACHA dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters


class SRIRACHAConfig(Config):
    """High-level Config using SRIRACHA default parameters."""

    def create_parameters(self):
        """Create SRIRACHA parameters for this config."""
        return SRIRACHAParameters()


__all__ = ['SRIRACHAConfig']
