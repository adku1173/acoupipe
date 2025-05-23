import numpy as np
import pytest
from acoular import ImportGrid, MicGeom, SteeringVector

from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic

mg = MicGeom(
    pos_total=np.array(
        [
            [-0.68526741, -0.7593943, -1.99918406, 0.08414458],
            [-0.60619132, 1.20374544, -0.27378946, -1.38583541],
            [0.32909911, 0.56201909, -0.24697204, -0.68677001],
        ]
    )
)


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


@pytest.fixture
def psa():
    psa_instance = PowerSpectraAnalytic(
        block_size=256,
        overlap='50%',
        sample_freq=51200,
        num_samples=51200,
        steer=SteeringVector(mics=mg, grid=ImportGrid(pos=np.random.normal(size=(3, 3)))),
    )
    nfft = psa_instance.fftfreq()
    psa_instance.Q = np.stack([np.eye(3, dtype=complex) * 0.1 for _ in range(nfft.shape[0])])
    psa_instance.noise = np.stack([np.eye(mg.num_mics, dtype=complex) for _ in range(nfft.shape[0])])
    return psa_instance


@pytest.mark.parametrize(
    'attr, func, value',
    [
        ('seed', change_seed, [False, False]),
        ('mode', change_mode, [False, False]),
        ('overlap', change_overlap, [False, False]),
        ('num_samples', change_num_samples, [False, False]),
        ('noise', change_noise, [False, False]),
        ('ind_low', change_indices, [False, False]),
        ('Q', change_Q, [True, False]),
        ('steer', change_steer, [True, False]),
    ],
)
def test_noise_cached_property(psa, attr, func, value):
    psa.mode = 'wishart'
    noise1 = psa._noise.copy()
    csm1 = psa.csm.copy()
    func(psa)
    noise2 = psa._noise.copy()
    csm2 = psa.csm.copy()
    assert np.allclose(noise1, noise2) == value[0]
    assert np.allclose(csm1, csm2) == value[1]


@pytest.mark.parametrize(
    'attr, func, value',
    [
        ('seed', change_seed, [False, False]),
        ('mode', change_mode, [False, False]),
        ('overlap', change_overlap, [False, False]),
        ('num_samples', change_num_samples, [False, False]),
        ('noise', change_noise, [True, False]),
        ('ind_low', change_indices, [False, False]),
        ('Q', change_Q, [False, False]),
        ('steer', change_steer, [True, False]),
    ],
)
def test_Q_cached_property(psa, attr, func, value):
    psa.mode = 'wishart'
    Q1 = psa._Q.copy()
    csm1 = psa.csm.copy()
    func(psa)
    Q2 = psa._Q.copy()
    csm2 = psa.csm.copy()
    assert np.allclose(Q1, Q2) == value[0]
    assert np.allclose(csm1, csm2) == value[1]


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
