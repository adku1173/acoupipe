"""Common constants for acoupipe tests.

This module centralizes constants that are used across multiple test files
to avoid duplication and ensure consistency.
"""

import numpy as np
from acoular import MicGeom

# Feature names that are implemented and tested
IMPLEMENTED_FEATURES = [
    'time_data',
    'csm',
    'csmtriu',
    'sourcemap',
    'eigmode',
    'spectrogram',
    'seeds',
    'idx',
    'loc',
    'source_strength_analytic',
    'source_strength_estimated',
    'noise_strength_analytic',
    'noise_strength_estimated',
    'f',
    'num',
    'targetmap_analytic',
    'targetmap_estimated',
]

# Common test modes
MODES = ['welch', 'analytic', 'wishart']

# Common test frequencies
FREQUENCIES = [None, 1000]

# Common test nums (fractional octave band parameters)
NUMS = [0, 3]

# Default start index for tests
START_IDX = 3

# Signal length for tests that need a specific duration
TEST_SIGNAL_LENGTH = 2.0

# Microphone geometry position array (used in multiple test files)
POS_TOTAL = np.array(
    [
        [-0.68526741, -0.7593943, -1.99918406, 0.08414458],
        [-0.60619132, 1.20374544, -0.27378946, -1.38583541],
        [0.32909911, 0.56201909, -0.24697204, -0.68677001],
    ]
)

# MicGeom instance built from POS_TOTAL (for convenience)
MIC_GEOM = MicGeom(pos_total=POS_TOTAL)
