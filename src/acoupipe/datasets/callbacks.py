"""Callback helpers for Dataset lifecycle functions."""

import inspect

_UNSET = object()


def call_with_supported_context(callback, *, parameters=_UNSET, sampler=_UNSET, data=_UNSET):
    """Call a Dataset callback with compatible context injection.

    This helper is used across Dataset lifecycle hooks (prepare/feature/cleanup)
    to support both modern and legacy callback signatures without forcing a
    single function prototype.

    Supported invocation styles
    ---------------------------
    1. Named-context callbacks
       The callback explicitly names one or more of the supported context
       arguments: ``parameters``, ``sampler``, ``data``.

       Example signatures:
       ``lambda parameters: ...``
       ``lambda sampler, data: ...``
       ``def f(parameters=None, data=None): ...``

       In this mode, only arguments that are both supplied to this helper and
       declared by the callback are passed as keyword arguments.

    2. Legacy positional callbacks
       Older code may use anonymous positional placeholders (e.g. ``lambda _``)
       instead of semantic parameter names. To remain backward compatible, this
       function fills missing *required positional* parameters with context
       values in a fixed order:

       ``sampler`` -> ``data`` -> ``parameters``

       This order mirrors historical usage in Dataset feature hooks where the
       first positional argument was typically the sampler.

    How the fallback works
    ----------------------
    - First, keyword arguments are collected for explicitly declared context
      names.
    - The callback signature is then partially bound with those keywords.
    - Any remaining required positional parameters
      (``POSITIONAL_ONLY`` or ``POSITIONAL_OR_KEYWORD`` without defaults)
      are counted.
    - Up to that many positional context candidates are prepended and the
      callback is invoked as ``callback(*args, **kwargs)``.

    This design allows expressive named signatures for new code while keeping
    existing positional callbacks functional during the migration to the
    parameter-centric Dataset API.
    """
    signature = inspect.signature(callback)
    kwargs = {}
    if parameters is not _UNSET and 'parameters' in signature.parameters:
        kwargs['parameters'] = parameters
    if sampler is not _UNSET and 'sampler' in signature.parameters:
        kwargs['sampler'] = sampler
    if data is not _UNSET and 'data' in signature.parameters:
        kwargs['data'] = data

    # Backward compatibility: legacy callbacks may use positional placeholders
    # (e.g. ``lambda _``) instead of context parameter names.
    positional_candidates = []
    if sampler is not _UNSET:
        positional_candidates.append(sampler)
    if data is not _UNSET:
        positional_candidates.append(data)
    if parameters is not _UNSET:
        positional_candidates.append(parameters)

    bound = signature.bind_partial(**kwargs)
    missing_positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
        and parameter.name not in bound.arguments
    ]
    args = positional_candidates[: len(missing_positional)]
    return callback(*args, **kwargs)
