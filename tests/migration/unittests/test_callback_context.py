"""Unit tests for dataset callback context injection."""

# ruff: noqa: S101

from acoupipe.datasets.base.callbacks import call_with_supported_context

import pytest


def test_callback_receives_declared_context_only():
    """Test that named callbacks receive only declared context values."""

    def callback(parameters, data):
        return parameters['x'] + data['y']

    result = call_with_supported_context(
        callback,
        parameters={'x': 1},
        sampler={'ignored': True},
        data={'y': 2},
        state={'ignored': True},
    )

    assert result == 3


def test_callback_without_named_context_raises_type_error():
    """Test that anonymous positional callbacks fail instead of guessing context."""

    def callback(value):
        return value

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'value'"):
        call_with_supported_context(callback, sampler={'s': 1}, data={'d': 2})


def test_callback_can_request_runtime_state():
    """Test that callbacks may explicitly request lazy runtime state."""

    def callback(state):
        return state['runtime']

    result = call_with_supported_context(callback, state={'runtime': 5})

    assert result == 5


def test_callback_without_context_arguments_runs():
    """Test that callbacks may ignore all available context."""

    def callback():
        return 'constant'

    result = call_with_supported_context(
        callback,
        parameters='parameters',
        sampler='sampler',
        data='data',
        state='state',
    )

    assert result == 'constant'
