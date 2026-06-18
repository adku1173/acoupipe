"""
Lightweight Dataset parameter objects.

This module provides classes for managing parameter sets used in dataset generation.
The main class is :class:`ParameterSet`, which allows storing user-defined name/value
pairs and exposing them as Python attributes. Specialized parameter classes for
specific dataset types are also provided.

.. autosummary::
    :toctree: generated/

    ParameterSet
    SyntheticParameters
    MIRACLEParameters
"""

import json
import keyword
import tomllib
from pathlib import Path


class ParameterSet:
    """
    Mutable Monte-Carlo parameter state with attribute-style access.

    The :class:`ParameterSet` class is intentionally a plain Python object rather than a
    Traits object. It stores user-defined name/value pairs and exposes each parameter
    as a normal Python attribute. This allows for convenient access to parameters using
    dot notation.

    In the context of Monte-Carlo simulations, parameters can be sampled from distributions
    before each dataset item is generated. The parameter object holds both sampled values
    (e.g., source strengths) and feature settings (e.g., center frequency, bandwidth) that
    control the simulation.

    Parameters
    ----------
    **parameters : :class:`dict`, optional
        Initial parameter name/value pairs to store. Each key in the dictionary will
        become an attribute of the ParameterSet instance.

    Attributes
    ----------
    _parameter_names : :class:`list`
        Internal list tracking the names of all parameters in the order they were added.

    See Also
    --------
    :class:`SyntheticParameters` : Subclass for synthetic dataset parameters.
    :class:`MIRACLEParameters` : Subclass for MIRACLE dataset parameters.
    :gallerydoc:`examples/high_level_monte_carlo_dataset` : High-level Monte-Carlo datasets
        with parameter sampling.

    Notes
    -----
    - Parameter names must be valid Python identifiers (cannot be keywords, cannot start
      with underscore, must be alphanumeric with underscores).
    - Parameter names cannot conflict with existing :class:`ParameterSet` attributes.
    - Parameters can be added dynamically after instantiation using :meth:`add_parameter`.
    - In Monte-Carlo simulations, parameters are typically sampled using the Dataset API
      (see :gallerydoc:`examples/high_level_monte_carlo_dataset`).

    Examples
    --------
    Create a ParameterSet with initial parameters:

    >>> params = ParameterSet(c=343.0, frequency=1000)
    >>> params.c
    343.0
    >>> params.frequency
    1000

    Add a new parameter dynamically:

    >>> params.add_parameter('distance', 1.5)
    ParameterSet(c=343.0, frequency=1000, distance=1.5)
    >>> params.distance
    1.5

    Convert to dictionary:

    >>> params.to_dict()
    {'c': 343.0, 'frequency': 1000, 'distance': 1.5}
    """

    def __init__(self, **parameters):
        """
        Initialize a new ParameterSet with the given parameters.

        Parameters
        ----------
        **parameters : :class:`dict`, optional
            Initial parameter name/value pairs to store. Each key in the dictionary will
            become an attribute of the ParameterSet instance.

        Examples
        --------
        >>> params = ParameterSet(speed_of_sound=343.0, temperature=20.0)
        >>> params.speed_of_sound
        343.0
        """
        object.__setattr__(self, '_parameter_names', [])
        for name, value in parameters.items():
            self.add_parameter(name, value)

    @classmethod
    def from_dict(cls, parameters):
        """
        Create a ParameterSet from a dictionary of name/value pairs.

        Parameters
        ----------
        parameters : :class:`dict`
            Dictionary containing parameter name/value pairs.

        Returns
        -------
        :class:`ParameterSet`
            A new ParameterSet instance with the parameters from the dictionary.

        Examples
        --------
        >>> params = ParameterSet.from_dict({'c': 343.0, 'rho': 1.2})
        >>> params.c
        343.0
        >>> params.rho
        1.2
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
        path : :class:`str` or :class:`pathlib.Path`
            Path to the JSON file containing parameter name/value pairs.

        Returns
        -------
        :class:`ParameterSet`
            A new ParameterSet instance with the parameters loaded from the JSON file.

        Raises
        ------
        :obj:`FileNotFoundError`
            If the specified JSON file does not exist.
        :obj:`json.JSONDecodeError`
            If the JSON file contains invalid JSON syntax.

        Notes
        -----
        The JSON file should contain a JSON object (dictionary) with parameter
        name/value pairs.

        Examples
        --------
        Assuming a JSON file ``params.json`` with content:

        .. code-block:: json

            {"c": 343.0, "frequency": 1000}

        >>> params = ParameterSet.from_json('params.json')  # doctest: +SKIP
        >>> params.c  # doctest: +SKIP
        343.0
        """
        with Path(path).open(encoding='utf-8') as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_toml(cls, path):
        """
        Load parameters from a TOML file.

        Parameters
        ----------
        path : :class:`str` or :class:`pathlib.Path`
            Path to the TOML file containing parameter name/value pairs.

        Returns
        -------
        :class:`ParameterSet`
            A new ParameterSet instance with the parameters loaded from the TOML file.

        Raises
        ------
        :obj:`FileNotFoundError`
            If the specified TOML file does not exist.
        :obj:`tomllib.TOMLDecodeError`
            If the TOML file contains invalid TOML syntax.

        Notes
        -----
        The TOML file should contain a TOML table with parameter name/value pairs.
        This method requires Python 3.11 or later, as it uses the built-in :mod:`tomllib`
        module.

        Examples
        --------
        Assuming a TOML file ``params.toml`` with content:

        .. code-block:: toml

            c = 343.0
            frequency = 1000

        >>> params = ParameterSet.from_toml('params.toml')  # doctest: +SKIP
        >>> params.c  # doctest: +SKIP
        343.0
        """
        with Path(path).open('rb') as file:
            return cls.from_dict(tomllib.load(file))

    def add_parameter(self, name, value=None):
        """
        Add a named parameter and expose it as an attribute.

        Parameters
        ----------
        name : :class:`str`
            Name of the parameter. Must be a valid Python identifier.
        value : :class:`object`, optional
            Value of the parameter. Defaults to :obj:`None`.

        Returns
        -------
        :class:`ParameterSet`
            Returns self to allow method chaining.

        Raises
        ------
        :obj:`ValueError`
            If the parameter name is not a valid Python identifier, is a Python keyword,
            starts with an underscore, or conflicts with an existing :class:`ParameterSet`
            attribute.

        Examples
        --------
        >>> params = ParameterSet()
        >>> params.add_parameter('speed', 343.0).add_parameter('distance', 1.5)
        ParameterSet(speed=343.0, distance=1.5)
        >>> params.speed
        343.0
        """
        self._validate_name(name)
        if name not in self._parameter_names:
            self._parameter_names.append(name)
        setattr(self, name, value)
        return self

    def to_dict(self):
        """
        Return the current parameter values as a plain dictionary.

        Returns
        -------
        :class:`dict`
            Dictionary containing all parameter name/value pairs in the order they were added.

        Examples
        --------
        >>> params = ParameterSet(c=343.0, frequency=1000)
        >>> params.to_dict()
        {'c': 343.0, 'frequency': 1000}
        """
        return {name: getattr(self, name) for name in self._parameter_names}

    def __repr__(self):
        """
        Return a string representation of the ParameterSet with all parameter values.

        Returns
        -------
        :class:`str`
            String representation showing the class name and all parameter name/value pairs.

        Examples
        --------
        >>> params = ParameterSet(c=343.0, frequency=1000)
        >>> repr(params)
        "ParameterSet(c=343.0, frequency=1000)"
        """
        parameters = ', '.join(f'{name}={getattr(self, name)!r}' for name in self._parameter_names)
        return f'{type(self).__name__}({parameters})'

    @classmethod
    def _validate_name(cls, name):
        """
        Validate that a parameter name is acceptable.

        Parameters
        ----------
        name : :class:`str`
            The parameter name to validate.

        Raises
        ------
        :obj:`ValueError`
            If the name is not a string, is not a valid Python identifier, is a Python
            keyword, starts with an underscore, or conflicts with an existing
            :class:`ParameterSet` attribute.

        Notes
        -----
        This is an internal validation method used by :meth:`add_parameter` and
        :meth:`__init__`.
        """
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
            msg = f'Invalid parameter name {name!r}.'
            raise ValueError(msg)
        if hasattr(cls, name):
            msg = f'Invalid parameter name {name!r}; it conflicts with a ParameterSet attribute.'
            raise ValueError(msg)


class SyntheticParameters(ParameterSet):
    """
    Lightweight physical scene parameters for Monte-Carlo DatasetSynthetic generation.

    This class extends :class:`ParameterSet` to provide default parameters specifically
    for generating synthetic datasets. It includes a speed of sound parameter and a
    sourcemap parameter following the Acoular naming convention for feature-specific
    parameters.

    In Monte-Carlo simulations, parameters such as source strengths or positions can be
    sampled from distributions before each synthetic dataset item is generated.

    Parameters
    ----------
    c : :class:`float`, optional
        Speed of sound in meters per second. Defaults to 343.0 m/s.
    sourcemap : :class:`ParameterSet`, optional
        ParameterSet containing parameters required for the sourcemap feature calculation.
        Following the Acoular convention, this allows access to parameters via the
        ``sourcemap.<parameter>`` pattern (e.g., ``sourcemap.c`` for the speed of sound
        used by the beamformer). If not provided, a default ParameterSet with c=343.0
        is created. Defaults to :obj:`None`.

    Attributes
    ----------
    c : :class:`float`
        Speed of sound in meters per second.
    sourcemap : :class:`ParameterSet`
        ParameterSet containing parameters for the sourcemap feature, following the
        ``<feature_name>.<parameter>`` naming convention.

    See Also
    --------
    :class:`ParameterSet` : Base class for parameter management.
    :class:`MIRACLEParameters` : Parameters for MIRACLE dataset generation.
    :gallerydoc:`examples/high_level_monte_carlo_dataset` : Example showing parameter
        sampling for Monte-Carlo simulations.

    Notes
    -----
    The ``sourcemap`` attribute follows Acoular's naming convention where feature-specific
    parameters are accessed via ``<feature_name>.<parameter>``. For example, ``sourcemap.c``
    refers to the speed of sound parameter used by the beamformer for the sourcemap feature.

    Parameters can be sampled in Monte-Carlo simulations using the Dataset API. See
    :gallerydoc:`examples/high_level_monte_carlo_dataset` for a complete example.

    Examples
    --------
    Create SyntheticParameters with default values:

    >>> params = SyntheticParameters()
    >>> params.c
    343.0
    >>> isinstance(params.sourcemap, ParameterSet)
    True
    >>> params.sourcemap.c
    343.0

    Create SyntheticParameters with custom speed of sound:

    >>> params = SyntheticParameters(c=340.0)
    >>> params.c
    340.0
    """

    def __init__(self, c=343.0, sourcemap=None):
        if sourcemap is None:
            sourcemap = ParameterSet(c=343.0)
        super().__init__(c=c, sourcemap=sourcemap)


class MIRACLEParameters(ParameterSet):
    """
    Analysis parameters for measured-SRIR DatasetMIRACLE generation.

    This class extends :class:`ParameterSet` to provide parameters specifically for
    generating MIRACLE (Measured Impulse Responses for Acoustic Characterisation in
    Large Enclosures) datasets. It includes a sourcemap parameter following the Acoular
    naming convention for feature-specific parameters.

    Parameters
    ----------
    sourcemap : :class:`ParameterSet`, optional
        ParameterSet containing parameters required for the sourcemap feature calculation.
        Following the Acoular convention, this allows access to parameters via the
        ``sourcemap.<parameter>`` pattern (e.g., ``sourcemap.c`` for the speed of sound
        used by the beamformer). The speed of sound (c) in this sourcemap defaults to
        :obj:`None` as it may be determined from the measurement data. If not provided,
        a default ParameterSet with c=None is created. Defaults to :obj:`None`.

    Attributes
    ----------
    sourcemap : :class:`ParameterSet`
        ParameterSet containing parameters for the sourcemap feature, following the
        ``<feature_name>.<parameter>`` naming convention. The speed of sound (c) defaults
        to :obj:`None` as it is typically determined from the measurement data.

    See Also
    --------
    :class:`ParameterSet` : Base class for parameter management.
    :class:`SyntheticParameters` : Parameters for synthetic dataset generation.
    :gallerydoc:`examples/high_level_monte_carlo_dataset` : Example showing parameter
        sampling for Monte-Carlo simulations.

    Notes
    -----
    The ``sourcemap`` attribute follows Acoular's naming convention where feature-specific
    parameters are accessed via ``<feature_name>.<parameter>``. For example, ``sourcemap.c``
    refers to the speed of sound parameter used by the beamformer for the sourcemap feature.
    In MIRACLE datasets, this value is typically :obj:`None` initially and determined from
    the actual measurement data.

    While MIRACLE datasets typically use measured data, parameters can still be sampled
    or varied in Monte-Carlo style simulations using the Dataset API. See
    :gallerydoc:`examples/high_level_monte_carlo_dataset` for details.

    Examples
    --------
    Create MIRACLEParameters with default values:

    >>> params = MIRACLEParameters()
    >>> isinstance(params.sourcemap, ParameterSet)
    True
    >>> params.sourcemap.c is None
    True

    Create MIRACLEParameters with a custom sourcemap:

    >>> custom_sourcemap = ParameterSet(c=343.0, temperature=20.0)
    >>> params = MIRACLEParameters(sourcemap=custom_sourcemap)
    >>> params.sourcemap.c
    343.0
    """

    def __init__(self, sourcemap=None):
        if sourcemap is None:
            sourcemap = ParameterSet(c=None)
        super().__init__(sourcemap=sourcemap)
