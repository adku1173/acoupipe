import acoular as ac
from acoupipe.datasets.experimental import DatasetMIRACLEConfig

from tests.constants import FREQUENCIES, IMPLEMENTED_FEATURES, MODES, NUMS, START_IDX, TEST_SIGNAL_LENGTH

import numpy as np
import pytest


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('f', FREQUENCIES)
@pytest.mark.parametrize('num', NUMS)
def test_values_correct(mode, feature, f, num, create_dataset, snapshot):
    """Test generate method of the datasets in single task mode."""
    if f is None and num != 0:
        pytest.skip('Invalid combination of f=None and num!=0')
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode=mode)
    gen = dataset.generate(
        split='training', progress_bar=False, size=10000, start_idx=START_IDX, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == START_IDX:
            break
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the first eigenmode
        pytest.skip('Eigenmode test skipped due to numerical rounding errors associated with the OS')
    snapshot.check(np.asarray(data[feature]), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', FREQUENCIES)
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


@pytest.mark.parametrize('mode', MODES)
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


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', FREQUENCIES)
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


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('num', NUMS)
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


@pytest.mark.parametrize('mode', MODES)
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


def test_miracle_test_fixture_defaults_to_d1(create_miracle_dataset, monkeypatch):
    monkeypatch.setattr(DatasetMIRACLEConfig, 'create_acoular_pipeline', lambda self: None)

    dataset = create_miracle_dataset()

    assert dataset.config.scenario == 'D1'


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', FREQUENCIES)
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

    dataset = create_miracle_dataset(full=False, mode=mode, mic_sig_noise=mic_sig_noise, max_nsources=1)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    # Use microphone index 0 (valid for 4-microphone test config)
    csm_psq = data['csm'][:, 0, 0].sum()
    if mode != 'analytic':
        noise_psq = data['noise_strength_estimated'][:, 0].sum()
        sig_psq = data['source_strength_estimated'][:].sum()
    else:
        noise_psq = data['noise_strength_analytic'][:, 0].sum()
        sig_psq = data['source_strength_analytic'][:].sum()
    if mode != 'analytic':
        assert csm_psq == pytest.approx(noise_psq + sig_psq, rel=1e-1, abs=1e-1)
    else:
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', [1000])
def test_miracle_sourcemap_max(mode, num, f, create_miracle_dataset):
    """A plausibility test. Tolerance is large -> loudspeaker not a perfect monopole."""
    features = ['sourcemap', 'source_strength_estimated', 'source_strength_analytic']
    dataset = create_miracle_dataset(full=False, mode=mode, max_nsources=1, mic_sig_noise=False)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    sourcemap_max = ac.L_p(data['sourcemap'].max())
    source_stength_estimated = ac.L_p(data['source_strength_estimated'].max())
    np.testing.assert_allclose(sourcemap_max, source_stength_estimated, atol=3e0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_miracle_values_correct(mode, feature, f, num, create_miracle_dataset, snapshot):
    """Test generate method of the datasets in single task mode."""
    if mode == 'analytic' and '_estimated' in feature:
        pytest.skip('Feature not supported in analytic mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_miracle_dataset(full=False, mode=mode, signal_length=TEST_SIGNAL_LENGTH)
    gen = dataset.generate(
        split='training', progress_bar=False, size=10000, start_idx=START_IDX, f=f, num=num, features=[feature]
    )
    while True:
        data = next(gen)
        if data['idx'] == START_IDX:
            break
    if (
        feature == 'eigmode'
    ):  # consists of very small values with numerical rounding errors that stem from the eigen-decomposition
        # we therefore just test the first eigenmode
        snapshot.check(np.asarray(data[feature][:, :, -1]), rtol=1e-5, atol=1e-7)
    else:
        snapshot.check(np.asarray(data[feature]), rtol=1e-5, atol=1e-6)
