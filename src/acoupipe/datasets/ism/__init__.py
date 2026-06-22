"""Synthetic ISM dataset package."""

__all__ = ['DatasetSyntheticISM', 'DatasetSyntheticISMConfig', 'SyntheticISMConfig', 'SyntheticISMParameters']


def __getattr__(name):
    if name == 'DatasetSyntheticISM':
        from acoupipe.datasets.ism.dataset import DatasetSyntheticISM  # noqa: PLC0415

        return DatasetSyntheticISM
    if name == 'DatasetSyntheticISMConfig':
        from acoupipe.datasets.ism.legacy_config import DatasetSyntheticISMConfig  # noqa: PLC0415

        return DatasetSyntheticISMConfig
    if name == 'SyntheticISMConfig':
        from acoupipe.datasets.ism.config import SyntheticISMConfig  # noqa: PLC0415

        return SyntheticISMConfig
    if name == 'SyntheticISMParameters':
        from acoupipe.datasets.ism.parameters import SyntheticISMParameters  # noqa: PLC0415

        return SyntheticISMParameters
    raise AttributeError(name)
