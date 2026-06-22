"""SRIRACHA dataset class."""

from acoupipe.datasets.miracle.dataset import DatasetMIRACLE


class DatasetSRIRACHA(DatasetMIRACLE):
    """SRIRACHA dataset using experimentally measured spatial room impulse responses."""

    def __init__(
        self,
        srir_dir=None,
        scenario='SR1-D',
        dataset_split=None,
        ref_mic_index=63,
        mode='welch',
        mic_sig_noise=True,
        random_signal_length=False,
        signal_length=5,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        config=None,
    ):
        """Initialize the DatasetSRIRACHA object."""
        if config is None:
            from acoupipe.datasets.sriracha.legacy_config import DatasetSRIRACHAConfig  # noqa: PLC0415

            config = DatasetSRIRACHAConfig(
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
        super().__init__(tasks=tasks, config=config)


__all__ = ['DatasetSRIRACHA']
