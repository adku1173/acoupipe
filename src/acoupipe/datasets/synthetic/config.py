"""Synthetic dataset configuration."""

from acoupipe.datasets.base.config import Config
from acoupipe.datasets.synthetic.parameters import SyntheticParameters


class SyntheticConfig(Config):
    """High-level Config using Synthetic default parameters."""

    def __init__(self, parameters=None, **traits):
        """Initialize SyntheticConfig with optional parameters."""
        if parameters is None:
            parameters = self.create_parameters()
        # If parameters is a dict, convert it to SyntheticParameters
        if isinstance(parameters, dict):
            parameters = SyntheticParameters(**parameters)
        super().__init__(parameters=parameters, **traits)

    def create_parameters(self):
        """Create Synthetic parameters for this config."""
        return SyntheticParameters()


__all__ = ['SyntheticConfig']
