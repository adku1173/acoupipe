import shutil
import tempfile
from pathlib import Path

import acoular as ac
import numpy as np
import pytest

from acoupipe.datasets.experimental import DatasetMIRACLE
from acoupipe.datasets.synthetic import DatasetSynthetic, DatasetSyntheticTestConfig

IMPLEMENTED_FEATURES = ['time_data', 'csm', 'csmtriu', 'sourcemap', 'eigmode', 'spectrogram'] + [
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
TEST_SIGNAL_LENGTH = 0.5

dirpath = Path(__file__).parent.absolute()
modes = ['welch', 'analytic', 'wishart']
frequencies = [None, 1000]
nums = [0, 3]
validation_data_path = Path(__file__).parent.absolute() / 'validation_data'
start_idx = 3
tasks = 2

# TODO: speed up tests


@pytest.fixture
def temp_dir():
    """Fixture to create and clean up a temporary directory."""
    test_dir = Path(tempfile.mkdtemp())
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def create_dataset():
    """Fixture to create a DatasetSynthetic instance."""

    def _create_dataset(full=False, tasks=1, **kwargs):
        if full:
            return DatasetSynthetic(tasks=tasks, **kwargs)
        config = DatasetSyntheticTestConfig(**kwargs)
        return DatasetSynthetic(config=config, tasks=tasks, **kwargs)

    return _create_dataset


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('f', frequencies)
@pytest.mark.parametrize('num', nums)
def test_values_correct(mode, feature, f, num, create_dataset):
    """Test generate method of the datasets in single task mode."""
    if f is None and num != 0:
        pytest.skip('Invalid combination of f=None and num!=0')
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode=mode)
    gen = dataset.generate(
        split='training', progress_bar=False, size=10000, start_idx=start_idx, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == start_idx:
            break
    test_data = np.load(validation_data_path / f'{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy')
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the first eigenmode
        pytest.skip('Eigenmode test skipped due to numerical rounding errors associated with the OS')
    else:
        np.testing.assert_allclose(data[feature], test_data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', ['sourcemap'])
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_multiprocessing_values_correct(mode, feature, f, num, create_dataset):
    """Test generate method of the datasets in multiprocessing mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode=mode, tasks=tasks)
    gen = dataset.generate(
        split='training', progress_bar=False, size=100, start_idx=1, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == start_idx:
            break
    test_data = np.load(validation_data_path / f'{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy')
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the strongest eigenmode
        np.testing.assert_allclose(data[feature][:, :, -1], test_data[:, :, -1], rtol=1e-5, atol=1e-7)
    else:
        np.testing.assert_allclose(data[feature], test_data, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
def test_save_h5(mode, feature, num, f, temp_dir, create_dataset):
    """Test saving data to HDF5 format."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode)
    dataset.save_h5(split='training', size=2, features=[feature], name=temp_dir / 'test.h5', progress_bar=False)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
def test_get_feature_shapes(mode, feature, num, f, create_dataset):
    """Test if the output of the get_feature_shapes method matches with the generated shapes.

    This test considers a fixed number of sources (varying numbers result in None type shapes, which
    cannot be compared with the generated shapes). Varying source numbers are implicitly tested in the
    test_get_tf_dataset method.
    """
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode)
    data = next(dataset.generate(f=f, num=num, features=[feature], split='training', size=1, progress_bar=False))
    feature_collection = dataset.get_feature_collection(features=[feature], f=f, num=num)
    desired_shape = feature_collection.feature_tf_shape_mapper[feature]
    for i in range(len(desired_shape)):
        if desired_shape[i] is not None:
            assert desired_shape[i] == data[feature].shape[i]


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize(
    'kwargs', [{'mic_pos_noise': True}, {'mic_pos_noise': False}, {'mic_sig_noise': True}, {'mic_sig_noise': False}]
)
def test_toggle_sampler(mode, kwargs, create_dataset):
    dataset = create_dataset(mode=mode, **kwargs)
    gen = dataset.generate(
        split='training',
        size=1,
        progress_bar=False,
        features=['csm', 'noise_strength_estimated', 'noise_strength_analytic'],
    )
    data = next(gen)
    assert 'csm' in data.keys()
    if dataset.config.mic_sig_noise is not False:
        if mode != 'analytic':
            assert data['noise_strength_estimated'].sum() > 0
        assert data['noise_strength_analytic'].sum() > 0
    else:
        if mode != 'analytic':
            assert data['noise_strength_estimated'].sum() == 0
        assert data['noise_strength_analytic'].sum() == 0


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
def test_csm_prmssq(mode, mic_sig_noise, num, f, create_dataset):
    features = [
        'csm',
        'source_strength_estimated',
        'source_strength_analytic',
        'noise_strength_estimated',
        'noise_strength_analytic',
    ]
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')

    dataset = create_dataset(full=True, mode=mode, mic_sig_noise=mic_sig_noise, mic_pos_noise=False)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    csm_psq = data['csm'][:, 63, 63]
    if mode != 'analytic':
        noise_psq = data['noise_strength_estimated'][:, 63]
        sig_psq = data['source_strength_estimated'].sum(1)
    else:
        noise_psq = data['noise_strength_analytic'][:, 63]
        sig_psq = data['source_strength_analytic'].sum(1)
    if mode != 'analytic':
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq, rtol=1e-1, atol=1e-1)
    else:
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('num', nums)
def test_sourcemap_max(mode, num, create_dataset):
    """A plausibility test."""
    features = ['sourcemap', 'source_strength_estimated', 'source_strength_analytic']
    f = 4000
    dataset = create_dataset(
        full=False, mode=mode, mic_sig_noise=False, mic_pos_noise=False, max_nsources=1, snap_to_grid=True
    )
    dataset.config.fft_params['block_size'] = 512
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    sourcemap_max = ac.L_p(data['sourcemap'].max())
    source_stength_estimated = ac.L_p(data['source_strength_estimated'].max())
    np.testing.assert_allclose(sourcemap_max, source_stength_estimated, atol=1e-1)


@pytest.mark.parametrize('mode', modes)
def test_eigvalsum_equal_csm(mode, create_dataset):
    """A plausibility test."""
    features = ['csm', 'eigmode']
    num = 0
    f = 4000
    dataset = create_dataset(
        full=True, mode=mode, mic_sig_noise=False, mic_pos_noise=False, max_nsources=1, snap_to_grid=True
    )
    dataset.config.fft_params['block_size'] = 512
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    eig, eigvec = np.linalg.eigh(data['csm'][0])
    eig_eig = np.linalg.norm(data['eigmode'][0], axis=0)
    np.testing.assert_allclose(eig_eig, np.abs(eig), rtol=1e-5, atol=1e-7)


@pytest.fixture
def create_miracle_dataset():
    """Fixture to create a DatasetMIRACLE instance."""

    def _create_dataset(tasks=1, **kwargs):
        return DatasetMIRACLE(tasks=tasks, **kwargs)

    return _create_dataset


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', frequencies)
def test_miracle_csm_prmssq(mode, mic_sig_noise, num, f, create_miracle_dataset):
    features = [
        'csm',
        'source_strength_estimated',
        'source_strength_analytic',
        'noise_strength_estimated',
        'noise_strength_analytic',
    ]
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')

    dataset = create_miracle_dataset(mode=mode, mic_sig_noise=mic_sig_noise, max_nsources=1)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    csm_psq = data['csm'][:, 63, 63].sum()
    if mode != 'analytic':
        noise_psq = data['noise_strength_estimated'][:, 63].sum()
        sig_psq = data['source_strength_estimated'][:].sum()
    else:
        noise_psq = data['noise_strength_analytic'][:, 63].sum()
        sig_psq = data['source_strength_analytic'][:].sum()
    if mode != 'analytic':
        assert csm_psq == pytest.approx(noise_psq + sig_psq, rel=1e-1, abs=1e-1)
    else:
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('num', nums)
@pytest.mark.parametrize('f', [1000])
def test_miracle_sourcemap_max(mode, num, f, create_miracle_dataset):
    """A plausibility test. Tolerance is large -> loudspeaker not a perfect monopole."""
    features = ['sourcemap', 'source_strength_estimated', 'source_strength_analytic']
    dataset = create_miracle_dataset(mode=mode, max_nsources=1, mic_sig_noise=False)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    sourcemap_max = ac.L_p(data['sourcemap'].max())
    source_stength_estimated = ac.L_p(data['source_strength_estimated'].max())
    np.testing.assert_allclose(sourcemap_max, source_stength_estimated, atol=3e0)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_miracle_values_correct(mode, feature, f, num, create_miracle_dataset):
    """Test generate method of the datasets in single task mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_miracle_dataset(mode=mode, signal_length=TEST_SIGNAL_LENGTH)
    gen = dataset.generate(
        split='training', progress_bar=False, size=10000, start_idx=start_idx, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == start_idx:
            break
    test_data = np.load(validation_data_path / f'{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy')
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the first eigenmode
        np.testing.assert_allclose(data[feature][:, :, -1], test_data[:, :, -1], rtol=1e-5, atol=1e-7)
    else:
        np.testing.assert_allclose(data[feature], test_data, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode', modes)
@pytest.mark.parametrize('feature', ['sourcemap'])
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_miracle_multiprocessing_values_correct(mode, feature, f, num, create_miracle_dataset):
    """Test generate method of the datasets in multiprocessing mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_miracle_dataset(mode=mode, signal_length=TEST_SIGNAL_LENGTH, tasks=tasks)
    gen = dataset.generate(
        split='training', progress_bar=False, size=100, start_idx=1, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == start_idx:
            break
    test_data = np.load(validation_data_path / f'{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy')
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the strongest eigenmode
        np.testing.assert_allclose(data[feature][:, :, -1], test_data[:, :, -1], rtol=1e-5, atol=1e-7)
    else:
        np.testing.assert_allclose(data[feature], test_data, rtol=1e-5, atol=1e-7)
