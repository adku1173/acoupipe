from acoular import ImportGrid

import numpy as np
import pytest
from numpy.random import default_rng


def change_seed(psa):
    psa.seed = psa.seed + 1
    return psa


def change_mode(psa):
    psa.mode = 'wishart' if psa.mode != 'wishart' else 'analytic'
    return psa


def change_overlap(psa):
    psa.overlap = '50%' if psa.overlap == 'None' else 'None'
    return psa


def change_num_samples(psa):
    psa.num_samples = psa.num_samples // 2
    return psa


def change_noise(psa):
    noise = psa.noise.copy()
    noise *= 10
    psa.noise = noise
    return psa


def change_indices(psa):
    psa.ind_low = psa.ind_low + 1
    return psa


def change_Q(psa):
    Q = psa.Q.copy()
    Q *= 10
    psa.Q = Q
    return psa


def change_steer(psa):
    psa.steer.grid = ImportGrid(pos=np.random.normal(size=(3, 3)))
    return psa


def change_window(psa):
    psa.window = 'Hanning'
    return psa


@pytest.mark.parametrize(
    'func',
    [
        change_seed,
        change_mode,
        change_overlap,
        change_num_samples,
        change_noise,
        change_indices,
        change_Q,
        change_steer,
        change_window,
    ],
)
def test_wishart_csm_cached_property(psa, func):
    """The Wishart CSM must be recalculated whenever one of its dependencies changes."""
    psa.mode = 'wishart'
    csm1 = psa.csm.copy()
    assert np.array_equal(csm1, psa.csm)  # unchanged inputs -> identical result
    func(psa)
    csm2 = psa.csm.copy()
    assert not np.allclose(csm1, csm2)


def test_wishart_spectra(psa):
    psa.mode = 'wishart'
    # assume equal spectra
    csm1 = psa.csm.copy()
    csm2 = psa.csm.copy()
    assert np.allclose(csm1, csm2)
    # assume different spectra
    psa.steer.grid = ImportGrid(pos=np.random.normal(size=(3, 3)))
    csm1 = csm2
    csm2 = psa.csm
    assert not np.allclose(csm1, csm2)
    psa.seed = 10
    csm1 = csm2
    csm2 = psa.csm
    assert not np.allclose(csm1, csm2)
    # assume different spectra
    nfft = psa.fftfreq()
    psa.noise = np.stack([np.eye(4, dtype=complex) * 0.1 for _ in range(nfft.shape[0])])
    csm1 = csm2
    csm2 = psa.csm
    assert not np.allclose(csm1, csm2)
    # assume different spectra
    psa.noise = None
    csm1 = csm2
    csm2 = psa.csm
    assert not np.allclose(csm1, csm2)


def _random_hpd_matrix(dim, rng):
    X = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return X @ X.conj().T + np.eye(dim)


def test_sample_wishart_moments(psa):
    r"""Check the first two moments of :math:`\alpha \hat{C} \sim \mathcal{CW}(\alpha, C)`.

    For the complex Wishart distribution, :math:`E[\hat{C}] = C` and
    :math:`E[|\hat{C}_{ij} - C_{ij}|^2] = C_{ii} C_{jj} / \alpha`.
    """
    psa.num_samples = 256 * 8  # small number of degrees of freedom -> large effect of a bias
    df = psa.df_eq
    rng = default_rng(0)
    C = _random_hpd_matrix(4, rng)
    samples = np.array([psa._sample_wishart(C, rng) for _ in range(20000)])
    np.testing.assert_allclose(samples.mean(0), C, rtol=0, atol=0.03 * np.abs(C).max())
    var = (np.abs(samples - C) ** 2).mean(0)
    var_expected = np.outer(C.diagonal().real, C.diagonal().real) / df
    np.testing.assert_allclose(var, var_expected, rtol=0.05)


def _sample_csm_at_index(psa, index, nsamples):
    psa.ind_low = index
    psa.ind_high = index + 1
    psa.mode = 'analytic'
    C = psa.csm[index].copy()
    psa.mode = 'wishart'
    samples = []
    for seed in range(nsamples):
        psa.seed = seed
        samples.append(psa.csm[index].copy())
    return C, np.array(samples)


def test_wishart_csm_includes_cross_terms(psa):
    r"""The sample CSM must follow :math:`\alpha \hat{C} \sim \mathcal{CW}(\alpha, H Q H^H + N)`.

    Sampling the source and noise CSM separately neglects the signal-noise cross terms and
    underestimates the variance of the off-diagonal elements.
    """
    psa.num_samples = 256 * 8
    df = psa.df_eq
    C, samples = _sample_csm_at_index(psa, index=20, nsamples=4000)
    np.testing.assert_allclose(samples.mean(0), C, rtol=0, atol=0.05 * np.abs(C).max())
    var = (np.abs(samples - C) ** 2).mean(0)
    var_expected = np.outer(C.diagonal().real, C.diagonal().real) / df
    np.testing.assert_allclose(var, var_expected, rtol=0.1)


def test_wishart_csm_without_noise(psa):
    """Without noise, the CSM is sampled in source space (C = H Q H^H is singular)."""
    psa.noise = None
    nsources = psa.Q.shape[1]
    C, samples = _sample_csm_at_index(psa, index=20, nsamples=2000)
    np.testing.assert_allclose(samples.mean(0), C, rtol=0, atol=0.05 * np.abs(C).max())
    # rank of each sample is limited by the number of sources
    eigvals = np.linalg.eigvalsh(samples)
    assert np.all(np.abs(eigvals[:, :-nsources]) < 1e-10 * eigvals[:, -1:])


@pytest.mark.parametrize('noise', [True, False])
def test_wishart_df_too_small(psa, noise):
    if not noise:
        psa.noise = None
    dim = psa.steer.mics.num_mics if noise else psa.Q.shape[1]
    psa.mode = 'wishart'
    psa.overlap = 'None'
    psa.num_samples = psa.block_size * (dim - 1)
    assert psa.df_eq <= dim - 1
    with pytest.raises(ValueError, match='degrees of freedom'):
        _ = psa.csm
