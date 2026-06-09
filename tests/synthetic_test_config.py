"""Test configuration for synthetic dataset.

This module contains test-specific configuration classes that should not be exposed
in the public API.
"""

import acoular as ac
import numpy as np

from acoupipe.datasets.synthetic import DatasetSyntheticConfig


class DatasetSyntheticTestConfig(DatasetSyntheticConfig):
    """Test configuration for synthetic dataset with simplified microphone geometry.

    This configuration uses a smaller, fixed microphone geometry for faster testing.
    """

    def create_mics(self):
        return ac.MicGeom(
            pos_total=np.array(
                [
                    [-0.68526741, -0.7593943, -1.99918406, 0.08414458],
                    [-0.60619132, 1.20374544, -0.27378946, -1.38583541],
                    [0.32909911, 0.56201909, -0.24697204, -0.68677001],
                ],
            ),
        )

    def create_grid(self):
        ap = np.round(self.mics.aperture, decimals=12)
        return ac.RectGrid(
            y_min=-0.5 * ap,
            y_max=0.5 * ap,
            x_min=-0.5 * ap,
            x_max=0.5 * ap,
            z=0.5 * ap,
            increment=1 / 5 * ap,
        )
