"""MIRACLE dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.miracle.parameters import MIRACLEParameters


class MIRACLEConfig(Config):
    """High-level Config using MIRACLE default parameters."""

    def create_parameters(self):
        """Create MIRACLE parameters for this config."""
        return MIRACLEParameters()


__all__ = ['MIRACLEConfig']
