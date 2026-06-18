"""Callback helpers for Dataset lifecycle functions."""

import inspect

_UNSET = object()


def call_with_supported_context(callback, *, parameters=_UNSET, sampler=_UNSET, data=_UNSET):
    """Call a callback with only the supplied Dataset context arguments it declares."""
    signature = inspect.signature(callback)
    kwargs = {}
    if parameters is not _UNSET and 'parameters' in signature.parameters:
        kwargs['parameters'] = parameters
    if sampler is not _UNSET and 'sampler' in signature.parameters:
        kwargs['sampler'] = sampler
    if data is not _UNSET and 'data' in signature.parameters:
        kwargs['data'] = data
    return callback(**kwargs)
