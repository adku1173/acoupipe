"""Synthetic ISM dataset class."""

from acoupipe.datasets.ism.ir import require_ir_support
from acoupipe.datasets.ism.legacy_config import DatasetSyntheticISMConfig
from acoupipe.datasets.synthetic.dataset import DatasetSynthetic


class DatasetSyntheticISM(DatasetSynthetic):
    """Unsupported developer-only dataset class for impulse-response-based synthetic scenes."""

    def __init__(
        self,
        mode='welch',
        mic_pos_noise=True,
        mic_sig_noise=True,
        snap_to_grid=False,
        random_signal_length=False,
        signal_length=5,
        fs=13720.0,
        min_nsources=1,
        max_nsources=10,
        rt60=2.0,
        tasks=1,
        remote_args=None,
        logger=None,
        config=None,
    ):
        """Initialize the DatasetSyntheticISM object."""
        require_ir_support()
        if config is None:
            config = DatasetSyntheticISMConfig(
                mode=mode,
                signal_length=signal_length,
                fs=fs,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                mic_pos_noise=mic_pos_noise,
                mic_sig_noise=mic_sig_noise,
                snap_to_grid=snap_to_grid,
                random_signal_length=random_signal_length,
                rt60=rt60,
            )
        super().__init__(config=config, tasks=tasks, remote_args=remote_args, logger=logger)


__all__ = ['DatasetSyntheticISM']
