"""Base classes for AcouPipe.

This module provides abstract base classes that form the foundation of the AcouPipe
architecture. These classes are not intended to be used directly, but to be subclassed
by classes that implement the actual functionality.

.. inheritance-diagram::
                acoupipe.base
    :top-classes:
                acoupipe.base.DataGenerator
                acoupipe.base.BaseSampler
    :parts: 1

.. autosummary::
    :toctree: generated/

    DataGenerator
    BaseSampler
"""

from abc import abstractmethod

from numpy.random import Generator, RandomState
from scipy.stats import _distn_infrastructure
from traits.api import ABCHasStrictTraits, Bool, Either, Instance, Trait


class DataGenerator(ABCHasStrictTraits):
    """Abstract base class that serves as a data generator.

    This class should not be used directly. It provides a common interface for all
    classes that generate data via the :meth:`result` method in a block-wise or
    sample-wise manner.
    """

    @abstractmethod
    def get_data(self):
        """Python generator that iteratively yields data set samples as a dictionary.

        This method needs to be implemented by derived classes.

        Returns
        -------
        dict
            Dictionary containing a sample of the data set
            {feature_name[key] : feature[values]}.
        """


class BaseSampler(ABCHasStrictTraits):
    """Base class that represents a random process.

    This class has no functionality and should not be used in practice.
    It provides a common interface for all classes that manipulate attributes of
    an instance or a list of instances according to a specified random distribution.
    """

    #: a list of instances which attributes are to be manipulated
    target = Trait(list, desc='the object instances which are manipulated by this class')

    #: the random variable specifying the random distribution
    random_var = Instance(_distn_infrastructure.rv_frozen, desc='instance of a random variable from scipy.stats module')

    #: the state of the random variable :attr:`random_var`
    random_state = Either(int, RandomState, Generator, desc='random state of the random variable')

    #: manages if the same sampled value is chosen for all objects in the :attr:`target` list
    #: if False, one value for each target is drawn
    equal_value = Bool(False, desc='manages if a single value is chosen for all targets')

    def rvs(self, size=1):
        """Random variable sampling (for internal use).

        Parameters
        ----------
        size : int, optional
            The size of the output array. Defaults to 1.

        Returns
        -------
        array-like
            Random values drawn from the random distribution.
        """
        return self.random_var.rvs(size=size, random_state=self.random_state)

    @abstractmethod
    def sample(self):
        """Utilizes :meth:`rvs` function to draw random values from the random distribution.

        This method needs to be implemented by derived classes to perform the actual
        sampling and assign values to the target instances.
        """
