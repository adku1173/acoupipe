"""Unit tests for the per-feature prepare lifecycle.

A :class:`~acoupipe.datasets.features.Feature` may define an optional prepare
callback. :meth:`~acoupipe.datasets.features.Feature.get_feature_func` wraps it
together with the calculation, so that the prepare callback runs immediately
before the calculation of its own feature. The tests in this module cover the
feature-level contract only; the wiring into the high-level ``Config`` is tested
separately.
"""

# ruff: noqa: S101

from acoupipe.datasets.base.parameters import ParameterSet
from acoupipe.datasets.features import Feature, combine_feature_funcs, create_feature

import numpy as np
import pytest
from traits.api import Any


class RecordingFeature(Feature):
    """Feature that records the calls of its prepare and calculation callbacks."""

    record = Any()

    def get_calc_func(self):
        """Return a calculation that records its call and the visible ``data`` keys."""

        def calc(parameters, data):
            self.record.append(f'calc:{self.name}:{parameters.marker}:{sorted(data)}')
            return {self.name: np.int32(len(data))}

        return calc


def make_recording_feature(name, record):
    """Return a :class:`RecordingFeature` whose prepare callback records its call."""

    def prepare(parameters, data):
        record.append(f'prepare:{name}:{parameters.marker}:{sorted(data)}')
        return {f'{name}_prepared': True}

    return RecordingFeature(name=name, dtype=np.int32, shape=(), record=record, prepare_func=prepare)


def test_feature_prepare_runs_immediately_before_own_calculation():
    """Test that each prepare runs directly before its own calculation.

    Prepare callbacks are not run as a separate batch before all calculations.
    The prepare output of a feature is visible to its own calculation and to all
    later features through ``data``.
    """
    record = []
    parameters = ParameterSet(marker='sample')
    feature_a = make_recording_feature('a', record)
    feature_b = make_recording_feature('b', record)

    feature_func = combine_feature_funcs(
        [feature_a.get_feature_func(parameters), feature_b.get_feature_func(parameters)],
    )
    result = feature_func(sampler={})

    assert result['a'] == np.int32(1)
    assert result['b'] == np.int32(3)
    assert record == [
        'prepare:a:sample:[]',
        "calc:a:sample:['a_prepared']",
        "prepare:b:sample:['a', 'a_prepared']",
        "calc:b:sample:['a', 'a_prepared', 'b_prepared']",
    ]


def test_feature_without_prepare_still_works():
    """Test that features without a prepare callback calculate as before."""
    feature = create_feature(
        lambda sampler: {'plain': np.float32(1.0)},  # noqa: ARG005
        name='plain',
        shape=(),
        dtype=np.float32,
    )

    assert feature.get_prepare_func() is None
    assert feature.get_feature_func()(sampler={}, data={}) == {'plain': np.float32(1.0)}


def test_non_dict_prepare_return_is_ignored():
    """Test that only dictionaries returned by prepare are merged into ``data``.

    The feature output remains the responsibility of the calculation.
    """
    data = {}
    feature = create_feature(
        lambda data: {'value': len(data)},
        name='value',
        shape=(),
        dtype=np.int64,
        prepare_func=lambda: 'ignored',
    )

    assert feature.get_feature_func()(sampler={}, data=data) == {'value': 0}
    assert data == {}


def test_feature_callbacks_receive_declared_context():
    """Test that prepare and calculation receive only the contexts they declare by name."""
    received = {}
    parameters = ParameterSet(c=343.0)

    def prepare(sampler, parameters):
        received['prepare'] = (sampler, parameters)

    def calc(parameters, data):
        received['calc'] = (parameters, data)
        return {'c': parameters.c}

    feature = create_feature(calc, name='c', shape=(), dtype=np.float64, prepare_func=prepare)
    data = {}
    sampler = {'key': 'sampler'}

    assert feature.get_feature_func(parameters)(sampler=sampler, data=data) == {'c': 343.0}
    assert received['prepare'] == (sampler, parameters)
    assert received['calc'] == (parameters, data)


def test_feature_object_callbacks_require_named_context():
    """Test that anonymous positional callback arguments are not guessed.

    Feature callbacks declare their contexts by name. A callback such as
    ``lambda _: ...`` fails instead of silently receiving the sampler.
    """
    feature = create_feature(lambda _: {'x': 1}, name='x', shape=(), dtype=np.int64)

    with pytest.raises(TypeError, match='missing 1 required positional argument'):
        feature.get_feature_func()(sampler={}, data={})


def test_parameters_are_not_injected_without_parameters():
    """Test that ``parameters`` is only injected if it is passed to ``get_feature_func``.

    The legacy API calls ``get_feature_func()`` without parameters. Callbacks
    that request ``parameters`` therefore fail there instead of receiving ``None``.
    """
    feature = create_feature(lambda parameters: {'x': parameters}, name='x', shape=(), dtype=np.int64)

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'parameters'"):
        feature.get_feature_func()(sampler={}, data={})


def test_old_style_get_feature_func_override_still_works():
    """Test that subclasses overriding ``get_feature_func`` keep working.

    Old-style subclasses replace the wrapper. They calculate as before and
    their ``prepare_func`` is not run.
    """

    class OldStyleFeature(Feature):
        def get_feature_func(self):
            return lambda sampler: {self.name: 1}  # noqa: ARG005

    feature = OldStyleFeature(name='old', prepare_func=lambda: pytest.fail('prepare must not run'))

    assert combine_feature_funcs([feature.get_feature_func()])(sampler={}) == {'old': 1}
