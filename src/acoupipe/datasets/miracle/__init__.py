"""MIRACLE dataset package."""

__all__ = ['DatasetMIRACLE', 'DatasetMIRACLEConfig', 'MIRACLEConfig', 'MIRACLEParameters']


def __getattr__(name):
    if name == 'DatasetMIRACLE':
        from acoupipe.datasets.miracle.dataset import DatasetMIRACLE  # noqa: PLC0415

        return DatasetMIRACLE
    if name == 'DatasetMIRACLEConfig':
        from acoupipe.datasets.miracle.legacy_config import DatasetMIRACLEConfig  # noqa: PLC0415

        return DatasetMIRACLEConfig
    if name == 'MIRACLEConfig':
        from acoupipe.datasets.miracle.config import MIRACLEConfig  # noqa: PLC0415

        return MIRACLEConfig
    if name == 'MIRACLEParameters':
        from acoupipe.datasets.miracle.parameters import MIRACLEParameters  # noqa: PLC0415

        return MIRACLEParameters
    raise AttributeError(name)
