"""Tests the random state of the Sampler class.

Current behaviour is, that the random_state argument of the Sampler class
overwrites the random state of the random variable. If no RandomState is given
to the Sampler class, the random state of the random variable will be used.

"""


import unittest

from acoupipe import BaseSampler
from numpy.random import RandomState
from scipy.stats import norm

SEED = 100
NVALUES = 100

class TestSamplerState(unittest.TestCase):

    def _version1(self):
        n = norm()
        n.random_state = 1 # should be overwritten
        bs = BaseSampler(random_var=n,random_state=SEED)
        return bs.rvs(NVALUES)

    def _version2(self):
        n = norm()
        n.random_state = 1 # should be overwritten
        bs = BaseSampler(random_var=n,random_state=RandomState(SEED))
        return bs.rvs(NVALUES)

    def _version3(self):
        n = norm()
        n.random_state = SEED
        bs = BaseSampler(random_var=n)
        return bs.rvs(NVALUES)

    def _version4(self):
        n = norm()
        n.random_state = RandomState(SEED)
        bs = BaseSampler(random_var=n)
        return bs.rvs(NVALUES)

    def test(self):
        """Test that all versions result in the same random numbers."""
        for j,cal in enumerate([self._version2,self._version3,self._version4]):
            assert (self._version1() == cal()).all()
            print(f"version 1 equal to version {j+2}")

if __name__ == "__main__":
    unittest.main()
