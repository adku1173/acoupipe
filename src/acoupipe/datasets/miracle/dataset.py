"""MIRACLE dataset class."""

from acoupipe.datasets.base import DatasetBase


class DatasetMIRACLE(DatasetBase):
    """MIRACLE dataset using experimentally measured spatial room impulse responses."""

    def __init__(
        self,
        srir_dir=None,
        scenario='A1',
        dataset_split=None,
        ref_mic_index=63,
        mode='welch',
        mic_sig_noise=True,
        random_signal_length=False,
        signal_length=5,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        remote_args=None,
        config=None,
    ):
        """Initialize the DatasetMIRACLE object."""
        if config is None:
            from acoupipe.datasets.miracle.legacy_config import DatasetMIRACLEConfig  # noqa: PLC0415

            config = DatasetMIRACLEConfig(
                mode=mode,
                random_signal_length=random_signal_length,
                signal_length=signal_length,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                srir_dir=srir_dir,
                scenario=scenario,
                dataset_split=dataset_split,
                ref_mic_index=ref_mic_index,
                mic_sig_noise=mic_sig_noise,
            )
        super().__init__(tasks=tasks, remote_args=remote_args, config=config)


__all__ = ['DatasetMIRACLE']
