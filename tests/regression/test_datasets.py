import acoular as ac
from acoupipe.datasets.features import get_tf_mappers
from acoupipe.datasets.miracle.legacy_config import DatasetMIRACLEConfig

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
    if mode != 'welch' and '_estimated' in feature:
        pytest.skip('Feature not supported in non-welch mode')
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
    if mode != 'welch' and '_estimated' in feature:
        pytest.skip('Feature not supported in non-welch mode')
    if mode != 'welch' and feature in ['spectrogram', 'time_data']:
        pytest.skip('Feature not supported in non-welch mode')

    dataset = create_dataset(mode)
    data = next(dataset.generate(f=f, num=num, features=[feature], split='training', size=1, progress_bar=False))
    _, feature_objects = dataset.get_features(features=[feature], f=f, num=num)
    _, shapes, _ = get_tf_mappers(feature_objects)
    desired_shape = shapes[feature]
    for i in range(len(desired_shape)):
        if desired_shape[i] is not None:
            assert desired_shape[i] == data[feature].shape[i]


@pytest.mark.parametrize('mode', ['analytic', 'wishart'])
@pytest.mark.parametrize('feature', ['source_strength_estimated', 'noise_strength_estimated', 'targetmap_estimated'])
def test_estimated_features_raise_in_non_welch_mode(mode, feature, create_dataset):
    dataset = create_dataset(mode=mode)
    with pytest.raises(ValueError, match='not possible with modes'):
        dataset.get_features(features=[feature], f=1000, num=0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize(
    'kwargs', [{'mic_pos_noise': True}, {'mic_pos_noise': False}, {'mic_sig_noise': True}, {'mic_sig_noise': False}]
)
def test_toggle_sampler(mode, kwargs, create_dataset):
    noise_features = ['noise_strength_analytic']
    if mode == 'welch':
        noise_features.append('noise_strength_estimated')
    dataset = create_dataset(mode=mode, **kwargs)
    gen = dataset.generate(
        split='training',
        size=1,
        progress_bar=False,
        features=['csm', *noise_features],
    )
    data = next(gen)
    assert 'csm' in data.keys()
    for feature in noise_features:
        if dataset.config.mic_sig_noise is not False:
            assert data[feature].sum() > 0
        else:
            assert data[feature].sum() == 0


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('mic_sig_noise', [True, False])
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', FREQUENCIES)
def test_csm_prmssq(mode, mic_sig_noise, num, f, create_dataset):
    strength_type = 'estimated' if mode == 'welch' else 'analytic'
    features = ['csm', f'source_strength_{strength_type}', f'noise_strength_{strength_type}']
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')

    dataset = create_dataset(full=True, mode=mode, mic_sig_noise=mic_sig_noise, mic_pos_noise=False)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    csm_psq = data['csm'][:, 63, 63]
    noise_psq = data[f'noise_strength_{strength_type}'][:, 63]
    sig_psq = data[f'source_strength_{strength_type}'].sum(1)
    if mode != 'analytic':
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq, rtol=1e-1, atol=1e-1)
    else:
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('num', NUMS)
def test_sourcemap_max(mode, num, create_dataset):
    """A plausibility test."""
    strength_type = 'estimated' if mode == 'welch' else 'analytic'
    features = ['sourcemap', f'source_strength_{strength_type}']
    f = 4000
    dataset = create_dataset(
        full=False, mode=mode, mic_sig_noise=False, mic_pos_noise=False, max_nsources=1, snap_to_grid=True
    )
    dataset.config.fft_params['block_size'] = 512
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    sourcemap_max = ac.L_p(data['sourcemap'].max())
    source_strength = ac.L_p(data[f'source_strength_{strength_type}'].max())
    # the Wishart-distributed CSM fluctuates around the analytic source strength
    atol = 5e-1 if mode == 'wishart' else 1e-1
    np.testing.assert_allclose(sourcemap_max, source_strength, atol=atol)


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
    strength_type = 'estimated' if mode == 'welch' else 'analytic'
    features = ['csm', f'source_strength_{strength_type}', f'noise_strength_{strength_type}']
    if num == 3 and f is None:
        pytest.skip('Invalid combination of num=3 and f=None')

    dataset = create_miracle_dataset(full=False, mode=mode, mic_sig_noise=mic_sig_noise, max_nsources=1)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    # Use microphone index 0 (valid for 4-microphone test config)
    csm_psq = data['csm'][:, 0, 0].sum()
    noise_psq = data[f'noise_strength_{strength_type}'][:, 0].sum()
    sig_psq = data[f'source_strength_{strength_type}'][:].sum()
    if mode != 'analytic':
        assert csm_psq == pytest.approx(noise_psq + sig_psq, rel=1e-1, abs=1e-1)
    else:
        np.testing.assert_allclose(csm_psq, noise_psq + sig_psq)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('num', NUMS)
@pytest.mark.parametrize('f', [1000])
def test_miracle_sourcemap_max(mode, num, f, create_miracle_dataset):
    """A plausibility test. Tolerance is large -> loudspeaker not a perfect monopole."""
    strength_type = 'estimated' if mode == 'welch' else 'analytic'
    features = ['sourcemap', f'source_strength_{strength_type}']
    dataset = create_miracle_dataset(full=False, mode=mode, max_nsources=1, mic_sig_noise=False)
    gen = dataset.generate(f=f, num=num, features=features, split='training', size=1, progress_bar=False)
    data = next(gen)
    sourcemap_max = ac.L_p(data['sourcemap'].max())
    source_strength = ac.L_p(data[f'source_strength_{strength_type}'].max())
    np.testing.assert_allclose(sourcemap_max, source_strength, atol=3e0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('feature', IMPLEMENTED_FEATURES)
@pytest.mark.parametrize('f', [1000])
@pytest.mark.parametrize('num', [0])
def test_miracle_values_correct(mode, feature, f, num, create_miracle_dataset, snapshot):
    """Test generate method of the datasets in single task mode."""
    if mode != 'welch' and '_estimated' in feature:
        pytest.skip('Feature not supported in non-welch mode')
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
