"""Test configuration for MIRACLE dataset.

This module contains test-specific configuration classes that should not be exposed
in the public API.
"""

import h5py as h5
import numpy as np

from acoupipe.datasets.experimental import DatasetMIRACLEConfig


class DatasetMIRACLETestConfig(DatasetMIRACLEConfig):
    """Test configuration for MIRACLE dataset with only 4 innermost microphones.

    This configuration uses only the 4 innermost microphones from the 64-microphone
    array for faster testing. The scenario is fixed to 'D1'.
    The grid parameters are similar to DatasetSyntheticTestConfig.
    """

    def __init__(self, **kwargs):
        # Force scenario to D1 for test config
        kwargs['scenario'] = 'D1'
        super().__init__(**kwargs)

    def create_mics(self):
        """Create microphone geometry with only the 4 innermost microphones."""
        import acoular as ac

        # Load all microphone positions from the file
        with h5.File(self.filename, 'r') as file:
            all_positions = file['data/location/receiver'][()].T

        # Select the 4 innermost microphones
        # For a planar array, these would be the 4 closest to the center
        # We'll select indices that form a small square in the center
        # For the 64-mic Vogel spiral, the innermost are typically the last few indices
        # Let's use a simple approach: select 4 mics closest to the geometric center
        center = np.mean(all_positions, axis=1)
        distances = np.linalg.norm(all_positions - center[:, np.newaxis], axis=0)
        innermost_indices = np.argsort(distances)[:4]

        # Sort indices to maintain consistent ordering
        innermost_indices = np.sort(innermost_indices)

        pos_total = all_positions[:, innermost_indices]
        return ac.MicGeom(pos_total=pos_total)

    def create_grid(self):
        """Create grid with parameters similar to DatasetSyntheticTestConfig."""
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

    def create_steer(self):
        """Create steering vector using one of the 4 microphones as reference."""
        import acoular as ac

        # Use the first microphone as reference (or any of the 4)
        # For a small array, the reference should be near the center
        ref_index = 0  # First of the 4 selected microphones
        return ac.SteeringVector(
            steer_type='true level',
            ref=self.mics.pos_total[:, ref_index],
            mics=self.mics,
            grid=self.grid,
            env=self.env,
        )
