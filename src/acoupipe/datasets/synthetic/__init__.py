"""Synthetic dataset package.

The package exposes the legacy synthetic dataset API for backwards
compatibility and the new high-level configuration API in package-local
modules.
"""

__all__ = [
    'DatasetSynthetic',
    'DatasetSyntheticConfig',
    'DatasetSyntheticISM',
    'DatasetSyntheticISMConfig',
    'SyntheticConfig',
    'SyntheticParameters',
    'sample_mic_noise_variance',
    'sample_rms',
    'sample_signal_length',
    'sample_signal_seed',
]


def __getattr__(name):
    if name == 'DatasetSynthetic':
        from acoupipe.datasets.synthetic.dataset import DatasetSynthetic  # noqa: PLC0415

        return DatasetSynthetic
    if name == 'DatasetSyntheticConfig':
        from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig  # noqa: PLC0415

        return DatasetSyntheticConfig
    if name == 'DatasetSyntheticISM':
        from acoupipe.datasets.ism.dataset import DatasetSyntheticISM  # noqa: PLC0415

        return DatasetSyntheticISM
    if name == 'DatasetSyntheticISMConfig':
        from acoupipe.datasets.ism.legacy_config import DatasetSyntheticISMConfig  # noqa: PLC0415

        return DatasetSyntheticISMConfig
    if name == 'SyntheticConfig':
        from acoupipe.datasets.synthetic.config import SyntheticConfig  # noqa: PLC0415

        return SyntheticConfig
    if name == 'SyntheticParameters':
        from acoupipe.datasets.synthetic.parameters import SyntheticParameters  # noqa: PLC0415

        return SyntheticParameters
    if name in {'sample_mic_noise_variance', 'sample_rms', 'sample_signal_length', 'sample_signal_seed'}:
        from acoupipe.datasets.synthetic import legacy_config  # noqa: PLC0415

        return getattr(legacy_config, name)
    raise AttributeError(name)
