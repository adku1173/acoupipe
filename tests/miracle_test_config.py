"""Test configuration for MIRACLE dataset.

This module contains test-specific configuration classes that should not be exposed
in the public API.
"""

from acoupipe.datasets.experimental import DatasetMIRACLEConfig

import numpy as np


class DatasetMIRACLETestConfig(DatasetMIRACLEConfig):
    """Test configuration for MIRACLE dataset with simplified microphone geometry.

    This configuration uses a small 4-microphone array for faster testing.
    The scenario is fixed to 'D1' with its parameters (speed of sound: 344.8 m/s).
    The grid parameters are similar to DatasetSyntheticTestConfig.

    Note: This test config bypasses the HDF5 file loading for the microphone geometry
    to ensure consistent behavior with only 4 microphones.
    """

    def __init__(self, **kwargs):
        # Force scenario to D1 for test config
        kwargs['scenario'] = 'D1'
        # Use microphone index 0 as reference (valid for 4-mic array)
        kwargs.setdefault('ref_mic_index', 0)
        # Disable positional noise for consistency
        kwargs.setdefault('mic_pos_noise', False)
        super().__init__(**kwargs)

    def set_filename(self):
        """Override to skip HDF5 file loading.

        Since we're using hardcoded microphone positions, we don't need the actual file.
        Set a dummy filename to avoid errors.
        """
        self._filename = 'dummy.h5'

    def create_env(self):
        """Create environment with D1 scenario's speed of sound."""
        import acoular as ac

        # D1 scenario has c0 = 344.8 m/s
        return ac.Environment(c=344.8)

    def create_mics(self):
        """Create microphone geometry with 4 microphones in a small planar arrangement."""
        import acoular as ac

        # Create a small 2x2 square array with 0.1m spacing
        pos_total = np.array(
            [
                [-0.05, -0.05, 0.05, 0.05],  # x positions
                [-0.05, 0.05, -0.05, 0.05],  # y positions
                [0.0, 0.0, 0.0, 0.0],  # z positions
            ]
        )
        return ac.MicGeom(pos_total=pos_total)

    def create_steer(self):
        """Create steering vector using the first microphone as reference."""
        import acoular as ac

        # Use the first microphone as reference
        ref_pos = self.mics.pos_total[:, 0]
        return ac.SteeringVector(
            steer_type='true level',
            ref=ref_pos,
            mics=self.mics,
            grid=self.grid,
            env=self.env,
        )

    def create_grid(self):
        """Create grid with parameters similar to DatasetSyntheticTestConfig."""
        import acoular as ac

        ap = self.mics.aperture
        # Use a z position that's reasonable for the array
        return ac.RectGrid(
            y_min=-0.5 * ap,
            y_max=0.5 * ap,
            x_min=-0.5 * ap,
            x_max=0.5 * ap,
            z=0.5 * ap,
            increment=1 / 5 * ap,
        )

    def create_source_grid(self):
        """Create source grid matching the observation area."""
        import acoular as ac

        ap = self.mics.aperture
        return ac.RectGrid(
            y_min=-0.5 * ap,
            y_max=0.5 * ap,
            x_min=-0.5 * ap,
            x_max=0.5 * ap,
            z=0.5 * ap,
            increment=1 / 5 * ap,
        )

    def create_sources(self):
        """Create sources - use PointSource instead of PointSourceConvolve for simplicity."""
        import acoular as ac

        sources = []
        for signal in self.signals:
            sources.append(
                ac.PointSource(
                    signal=signal,
                    mics=self.noisy_mics,
                    env=self.env,
                ),
            )
        return sources

    def get_prepare_func(self):
        """Override to use DatasetSyntheticConfig's prepare functions."""
        from functools import partial

        from acoupipe.datasets.synthetic import DatasetSyntheticConfig

        cf = DatasetSyntheticConfig
        if self.mode == 'welch':
            prepare_func = partial(
                cf.calc_welch_prepare_func,
                mics=self.mics,
                beamformer=self.beamformer,
                sources=self.sources,
                source_steer=self.source_steer,
                fft_spectra=self.fft_spectra,
                fft_obs_spectra=self.fft_obs_spectra,
                obs=self.obs,
            )
        else:
            prepare_func = partial(
                cf.calc_analytic_prepare_func,
                mics=self.mics,
                freq_data=self.freq_data,
            )
        return prepare_func
