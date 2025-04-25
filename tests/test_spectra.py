import unittest

import numpy as np
from acoular import ImportGrid, MicGeom, SteeringVector
from parameterized import parameterized

from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic

mg = MicGeom(pos_total=np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]]))

def change_seed(psa):
    psa.seed = psa.seed +1
    return psa

def change_mode(psa):
    if psa.mode != "wishart":
        psa.mode = "wishart"
    else:
        psa.mode = "analytic"
    return psa

def change_block_size(psa):
    psa.block_size = psa.block_size // 2
    return psa

def change_overlap(psa):
    if psa.overlap == "None":
        psa.overlap = "50%"
    else:
        psa.overlap = "None"
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
    psa.steer.grid = ImportGrid(gpos_file=np.random.normal(size=(3,3)))
    return psa

class TestPowerSpectraAnalytic(unittest.TestCase):

    def setUp(self):
        self.psa = PowerSpectraAnalytic(
            block_size = 256,
            overlap = "50%",
            sample_freq = 51200,
            num_samples = 51200,
            steer=SteeringVector(mics=mg, grid=ImportGrid(gpos_file=np.random.normal(size=(3,3)))))
        nfft = self.psa.fftfreq()
        self.psa.Q = np.stack([np.eye(3,dtype=complex)*0.1 for _ in range(nfft.shape[0])])
        self.psa.noise = np.stack([np.eye(mg.num_mics,dtype=complex) for _ in range(nfft.shape[0])])

    @parameterized.expand(
        [
            ("seed", change_seed, [False,False]),
            ("mode", change_mode, [False,False]),
            ("overlap", change_overlap, [False,False]),
            ("num_samples", change_num_samples, [False,False]),
            ("noise", change_noise, [False,False]),
            ("ind_low", change_indices, [False,False]),
            ("Q", change_Q, [True,False]),
            ("steer", change_steer, [True,False])
            # ("block_size", change_block_size),
        ]
    )
    def test_noise_cached_property(self, attr, func, value):
        with self.subTest(attr=attr):
            self.psa.mode = "wishart"
            noise1 = self.psa._noise.copy()
            csm1 = self.psa.csm.copy()
            func(self.psa)
            noise2 = self.psa._noise.copy()
            csm2 = self.psa.csm.copy()
            self.assertEqual(np.allclose(noise1, noise2), value[0])
            self.assertEqual(np.allclose(csm1, csm2), value[1])

    @parameterized.expand(
        [
            ("seed", change_seed, [False,False]),
            ("mode", change_mode, [False,False]),
            ("overlap", change_overlap, [False,False]),
            ("num_samples", change_num_samples, [False,False]),
            ("noise", change_noise, [True,False]),
            ("ind_low", change_indices, [False,False]),
            ("Q", change_Q, [False,False]),
            ("steer", change_steer, [True,False])
            # ("block_size", change_block_size),
        ]
    )
    def test_Q_cached_property(self, attr, func, value):
        with self.subTest(attr=attr):
            self.psa.mode = "wishart"
            Q1 = self.psa._Q.copy()
            csm1 = self.psa.csm.copy()
            func(self.psa)
            Q2 = self.psa._Q.copy()
            csm2 = self.psa.csm.copy()
            self.assertEqual(np.allclose(Q1, Q2), value[0])
            self.assertEqual(np.allclose(csm1, csm2), value[1])

    def test_wishart_spectra(self):
        self.psa.mode = "wishart"
        # assume equal spectra
        csm1 = self.psa.csm.copy()
        csm2 = self.psa.csm.copy()
        self.assertTrue(np.allclose(csm1, csm2))
        # assume different spectra
        self.psa.steer.grid = ImportGrid(gpos_file=np.random.normal(size=(3,3)))
        csm1 = csm2
        csm2 = self.psa.csm
        self.assertFalse(np.allclose(csm1, csm2))
        self.psa.seed = 10
        csm1 = csm2
        csm2 = self.psa.csm
        self.assertFalse(np.allclose(csm1, csm2))
        # assume different spectra
        nfft = self.psa.fftfreq()
        self.psa.noise = np.stack([np.eye(4,dtype=complex)*0.1 for _ in range(nfft.shape[0])])
        csm1 = csm2
        csm2 = self.psa.csm
        self.assertFalse(np.allclose(csm1, csm2))
        # assume different spectra
        self.psa.noise = None
        csm1 = csm2
        csm2 = self.psa.csm
        self.assertFalse(np.allclose(csm1, csm2))


if __name__ == "__main__":
    unittest.main()

