import os

import numpy as np
import pytest

from tests.constants import MODES, START_IDX, TEST_SIGNAL_LENGTH

# tasks is defined locally in this file
tasks = 2


pytestmark = [
    pytest.mark.multiprocessing,
    pytest.mark.skipif(os.environ.get('CI') == 'true', reason='Skip multiprocessing tests in CI'),
]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', ['sourcemap'])
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_multiprocessing_values_correct(mode, feature, f, num, create_dataset, snapshot):
    """Test generate method of the datasets in multiprocessing mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode=mode, tasks=tasks)
    gen = dataset.generate(
        split='training', progress_bar=False, size=100, start_idx=START_IDX, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == START_IDX:
            break
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the strongest eigenmode
        snapshot.check(np.asarray(data[feature][:, :, -1]), rtol=1e-5, atol=1e-7)
    else:
        snapshot.check(np.asarray(data[feature]), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', ['sourcemap'])
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_miracle_multiprocessing_values_correct(mode, feature, f, num, create_miracle_dataset, snapshot):
    """Test generate method of the datasets in multiprocessing mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_miracle_dataset(full=False, mode=mode, signal_length=TEST_SIGNAL_LENGTH, tasks=tasks)
    gen = dataset.generate(
        split='training', progress_bar=False, size=100, start_idx=START_IDX, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == START_IDX:
            break
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the strongest eigenmode
        snapshot.check(np.asarray(data[feature][:, :, -1]), rtol=1e-5, atol=1e-7)
    else:
        snapshot.check(np.asarray(data[feature]), rtol=1e-5, atol=1e-6)
