import unittest

import numpy as np
from acoular import ImportGrid, MicGeom, SteeringVector
from acoupipe import PowerSpectraAnalytic

mg = MicGeom(mpos_tot=np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]]))

class TestPowerSpectraAnalytic(unittest.TestCase):

    def setUp(self):
        self.psa = PowerSpectraAnalytic(steer=SteeringVector(mics=mg, grid=ImportGrid(gpos_file=np.random.normal(size=(3,3)))))
        self.psa.Q = 0.1*np.eye(3)[np.newaxis]
        self.psa.frequencies = np.array([1000])

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
        self.psa.noise = np.eye(4)[np.newaxis]
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

