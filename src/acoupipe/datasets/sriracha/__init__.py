"""SRIRACHA dataset package."""

__all__ = ['DatasetSRIRACHA', 'DatasetSRIRACHAConfig', 'SRIRACHAConfig', 'SRIRACHAParameters']


def __getattr__(name):
    if name == 'DatasetSRIRACHA':
        from acoupipe.datasets.sriracha.dataset import DatasetSRIRACHA  # noqa: PLC0415

        return DatasetSRIRACHA
    if name == 'DatasetSRIRACHAConfig':
        from acoupipe.datasets.sriracha.legacy_config import DatasetSRIRACHAConfig  # noqa: PLC0415

        return DatasetSRIRACHAConfig
    if name == 'SRIRACHAConfig':
        from acoupipe.datasets.sriracha.config import SRIRACHAConfig  # noqa: PLC0415

        return SRIRACHAConfig
    if name == 'SRIRACHAParameters':
        from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters  # noqa: PLC0415

        return SRIRACHAParameters
    raise AttributeError(name)
