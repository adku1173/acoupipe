"""Callback helpers for Dataset lifecycle functions."""

import inspect

_CONTEXT_NAMES = ('parameters', 'sampler', 'data', 'state')
_UNSET = object()


def call_with_supported_context(callback, *, parameters=_UNSET, sampler=_UNSET, data=_UNSET, state=_UNSET):
    """Call a Dataset callback with explicitly declared context values.

    The callback receives only context arguments that are both supplied to this
    helper and declared by name in the callback signature. Anonymous positional
    arguments are not guessed. Low-level Feature-function compatibility should
    be handled at the low-level Feature object boundary instead.

    Parameters
    ----------
    callback : callable
        Function to call.
    parameters : object, optional
        Dataset parameter container to inject when requested.
    sampler : object, optional
        Sampler context to inject when requested.
    data : object, optional
        Current sample data to inject when requested.
    state : object, optional
        Runtime state context to inject when requested.

    Returns
    -------
    object
        The value returned by ``callback``.

    Examples
    --------
    Named callbacks receive the matching context by keyword:

    >>> call_with_supported_context(lambda parameters: parameters['x'], parameters={'x': 1})
    1

    Callbacks may ignore all context:

    >>> call_with_supported_context(lambda: 'constant', sampler='sampler')
    'constant'
    """
    signature = inspect.signature(callback)
    context = _provided_context(parameters=parameters, sampler=sampler, data=data, state=state)
    kwargs = _declared_context_kwargs(signature, context)
    return callback(**kwargs)


def _provided_context(**context):
    """Return supplied context values in the supported injection order.

    ``_UNSET`` is the private sentinel for "this context was not supplied".
    It is different from ``None``: ``None`` may be an intentional context
    value and must still be forwarded to callbacks that request it by name.

    For example, if ``state=None`` is passed to
    :func:`call_with_supported_context`, ``state`` remains present in the
    returned context mapping. If ``state`` is left at its default ``_UNSET``,
    it is removed here before callback-signature matching happens.
    """
    return {name: context[name] for name in _CONTEXT_NAMES if context[name] is not _UNSET}


def _declared_context_kwargs(signature, context):
    """Return context kwargs explicitly declared by the callback signature.

    ``context`` contains the supplied context objects, such as ``parameters``
    or ``data``. This helper keeps only entries whose names can be passed as
    keyword arguments to the inspected callback ``signature``. The returned
    dictionary contains references to the original context objects; it does not
    copy potentially large ``parameters``, ``sampler``, ``data``, or ``state``
    values.
    """
    return {name: value for name, value in context.items() if _accepts_keyword_context(signature, name)}


def _accepts_keyword_context(signature, name):
    """Return whether ``signature`` can receive ``name`` as a keyword argument."""
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
