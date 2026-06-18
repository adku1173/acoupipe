"""Lightweight Dataset parameter objects."""

import json
import keyword
import tomllib
from pathlib import Path


class ParameterSet:
    """Mutable Monte-Carlo parameter state with attribute-style access.

    ``ParameterSet`` is intentionally a plain Python object rather than a Traits
    object. It stores user-defined name/value pairs and exposes each parameter
    as a normal Python attribute.
    """

    def __init__(self, **parameters):
        object.__setattr__(self, '_parameter_names', [])
        for name, value in parameters.items():
            self.add_parameter(name, value)

    @classmethod
    def from_dict(cls, parameters):
        """Create parameters from a name/value mapping."""
        parameter_set = cls()
        for name, value in parameters.items():
            parameter_set.add_parameter(name, value)
        return parameter_set

    @classmethod
    def from_json(cls, path):
        """Load parameters from a JSON name/value mapping."""
        with Path(path).open(encoding='utf-8') as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_toml(cls, path):
        """Load parameters from a TOML name/value mapping."""
        with Path(path).open('rb') as file:
            return cls.from_dict(tomllib.load(file))

    def add_parameter(self, name, value=None):
        """Add a named parameter and expose it as an attribute."""
        self._validate_name(name)
        if name not in self._parameter_names:
            self._parameter_names.append(name)
        setattr(self, name, value)
        return self

    def to_dict(self):
        """Return the current parameter values as a plain dictionary."""
        return {name: getattr(self, name) for name in self._parameter_names}

    def __repr__(self):
        """Return a string representation of the ParameterSet with all parameter values."""
        parameters = ', '.join(f'{name}={getattr(self, name)!r}' for name in self._parameter_names)
        return f'{type(self).__name__}({parameters})'

    @classmethod
    def _validate_name(cls, name):
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
            msg = f'Invalid parameter name {name!r}.'
            raise ValueError(msg)
        if hasattr(cls, name):
            msg = f'Invalid parameter name {name!r}; it conflicts with a ParameterSet attribute.'
            raise ValueError(msg)


class SyntheticParameters(ParameterSet):
    """Lightweight physical scene parameters for Monte-Carlo DatasetSynthetic generation."""

    def __init__(self, c=343.0, sourcemap=None):
        if sourcemap is None:
            sourcemap = ParameterSet(c=343.0)
        super().__init__(c=c, sourcemap=sourcemap)


class MIRACLEParameters(ParameterSet):
    """Analysis parameters for measured-SRIR DatasetMIRACLE generation."""

    def __init__(self, sourcemap=None):
        if sourcemap is None:
            sourcemap = ParameterSet(c=None)
        super().__init__(sourcemap=sourcemap)
