"""
Lightweight dataset parameter objects.

This module provides plain Python containers for storing dataset-generation
parameters with attribute-style access. The generic :class:`ParameterSet` class
is intentionally independent of Traits so that sampled Monte-Carlo values can be
assigned and serialized without trait validation overhead. Dataset-specific
subclasses are defined in dataset-specific packages and re-exported from
:mod:`acoupipe.datasets` for convenience.

.. autosummary::
    :toctree: generated/

    ParameterSet
"""

import json
import keyword
import tomllib
from pathlib import Path


class ParameterSet:
    """
    Mutable parameter state with attribute-style access.

    A :class:`ParameterSet` stores user-defined name/value pairs and exposes
    each parameter as a normal Python attribute. It is useful for high-level
    dataset APIs where parameters are sampled before each generated item and can
    also be requested as zero-processing output features.

    Parameter names must be valid public Python identifiers. They may not be
    Python keywords, may not start with an underscore, and may not conflict with
    existing :class:`ParameterSet` attributes.

    Parameters
    ----------
    **parameters
        Initial parameter name/value pairs. Each key becomes a public attribute
        of the instance.

    Attributes
    ----------
    _parameter_names : list of str
        Names of tracked parameters in insertion order.

    Examples
    --------
    Create a parameter set with initial parameters:

    >>> parameters = ParameterSet(c=343.0, source_count=1)
    >>> parameters.c
    343.0
    >>> parameters.source_count
    1

    Add another parameter and convert the set back to a dictionary:

    >>> parameters.add_parameter('frequency', 1000)
    ParameterSet(c=343.0, source_count=1, frequency=1000)
    >>> parameters.to_dict()
    {'c': 343.0, 'source_count': 1, 'frequency': 1000}
    """

    def __init__(self, **parameters):
        """
        Initialize a parameter set.

        Parameters
        ----------
        **parameters
            Initial parameter name/value pairs. Each key is validated and then
            exposed as an attribute.
        """
        object.__setattr__(self, '_parameter_names', [])
        for name, value in parameters.items():
            self.add_parameter(name, value)

    @classmethod
    def from_dict(cls, parameters):
        """
        Create a parameter set from a dictionary.

        Parameters
        ----------
        parameters : dict
            Mapping of parameter names to values.

        Returns
        -------
        ParameterSet
            New parameter set containing the dictionary entries.

        Examples
        --------
        >>> parameters = ParameterSet.from_dict({'c': 343.0})
        >>> parameters.c
        343.0
        """
        parameter_set = cls()
        for name, value in parameters.items():
            parameter_set.add_parameter(name, value)
        return parameter_set

    @classmethod
    def from_json(cls, path):
        """
        Load parameters from a JSON file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a JSON file containing an object that maps parameter names
            to values.

        Returns
        -------
        ParameterSet
            New parameter set loaded from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        json.JSONDecodeError
            If the file does not contain valid JSON.
        """
        with Path(path).open(encoding='utf-8') as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_toml(cls, path):
        """
        Load parameters from a TOML file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file containing parameter name/value pairs.

        Returns
        -------
        ParameterSet
            New parameter set loaded from the TOML file.

        Raises
        ------
        FileNotFoundError
            If the TOML file does not exist.
        tomllib.TOMLDecodeError
            If the file does not contain valid TOML.
        """
        with Path(path).open('rb') as file:
            return cls.from_dict(tomllib.load(file))

    def add_parameter(self, name, value=None):
        """
        Add a named parameter and expose it as an attribute.

        Parameters
        ----------
        name : str
            Parameter name. It must be a valid public Python identifier and must
            not conflict with an existing :class:`ParameterSet` attribute.
        value : object, optional
            Parameter value. Defaults to :obj:`None`.

        Returns
        -------
        ParameterSet
            The instance itself, enabling method chaining.

        Raises
        ------
        ValueError
            If `name` is not a valid parameter name.
        """
        self._validate_name(name)
        if name not in self._parameter_names:
            self._parameter_names.append(name)
        setattr(self, name, value)
        return self

    def to_dict(self):
        """
        Return all tracked parameters as a dictionary.

        Returns
        -------
        dict
            Parameter name/value pairs in insertion order.
        """
        return {name: getattr(self, name) for name in self._parameter_names}

    def __repr__(self):
        """
        Return a representation containing all tracked parameter values.

        Returns
        -------
        str
            String representation of the parameter set.
        """
        parameters = ', '.join(f'{name}={getattr(self, name)!r}' for name in self._parameter_names)
        return f'{type(self).__name__}({parameters})'

    @classmethod
    def _validate_name(cls, name):
        """
        Validate a parameter name.

        Parameters
        ----------
        name : object
            Candidate parameter name.

        Raises
        ------
        ValueError
            If `name` is not a valid public Python identifier or conflicts with
            an existing :class:`ParameterSet` attribute.
        """
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
            msg = f'Invalid parameter name {name!r}.'
            raise ValueError(msg)
        if hasattr(cls, name):
            msg = f'Invalid parameter name {name!r}; it conflicts with a ParameterSet attribute.'
            raise ValueError(msg)


__all__ = ['ParameterSet']
