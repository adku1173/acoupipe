"""Tests the random state of the Sampler class.

Current behaviour is, that the random_state argument of the Sampler class
overwrites the random state of the random variable. If no RandomState is given
to the Sampler class, the random state of the random variable will be used.

"""

import unittest

from numpy.random import RandomState
from scipy.stats import norm

from acoupipe.base import BaseSampler
from acoupipe.sampler import NumericAttributeSampler

SEED = 100
NVALUES = 100


# Create a simple concrete sampler for testing
class SimpleSampler(BaseSampler):
    """Simple sampler for testing base class functionality."""

    #: attribute of the object in the :attr:`target` list that should be
    #: sampled by the random variable
    attribute = 'test_attr'

    def rvs(self, size=1):
        """Random variable sampling."""
        return self.random_var.rvs(size=size, random_state=self.random_state)

    def sample(self):
        """Sample implementation."""
        self.rvs()


class TestSamplerState(unittest.TestCase):
    def _version1(self):
        n = norm()
        n.random_state = 1  # should be overwritten
        bs = SimpleSampler(random_var=n, random_state=SEED, target=[])
        return bs.rvs(NVALUES)

    def _version2(self):
        n = norm()
        n.random_state = 1  # should be overwritten
        bs = SimpleSampler(random_var=n, random_state=RandomState(SEED), target=[])
        return bs.rvs(NVALUES)

    def _version3(self):
        n = norm()
        n.random_state = SEED
        bs = SimpleSampler(random_var=n, target=[])
        return bs.rvs(NVALUES)

    def _version4(self):
        n = norm()
        n.random_state = RandomState(SEED)
        bs = SimpleSampler(random_var=n, target=[])
        return bs.rvs(NVALUES)

    def test(self):
        """Test that all versions result in the same random numbers."""
        for j, cal in enumerate([self._version2, self._version3, self._version4]):
            assert (self._version1() == cal()).all()
            print(f'version 1 equal to version {j + 2}')


if __name__ == '__main__':
    unittest.main()
