"""Synthetic ISM dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.ism.parameters import SyntheticISMParameters


class SyntheticISMConfig(Config):
    """High-level Config using Synthetic ISM default parameters."""

    def create_parameters(self):
        """Create Synthetic ISM parameters for this config."""
        return SyntheticISMParameters()


__all__ = ['SyntheticISMConfig']
