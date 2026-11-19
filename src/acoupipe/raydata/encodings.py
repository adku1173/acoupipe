"""Encode feature values as Arrow-compatible columns of a :class:`ray.data.Dataset`."""

import numpy as np


def split_complex(data):
    """Replace each complex value by its real and imaginary parts ``<name>_real``, ``<name>_imag``.

    Arrow, and thus Ray Data, has no complex data type.
    """
    split = {}
    for key, value in data.items():
        if np.iscomplexobj(value):
            split[f'{key}_real'] = np.real(value)
            split[f'{key}_imag'] = np.imag(value)
        else:
            split[key] = value
    return split


def stack_values(values):
    """Stack per-sample values along a new first axis.

    Values of different shapes, e.g. source locations of a varying number of sources, are stored
    in an object array that Ray Data converts to a variable-shaped tensor column.
    """
    if len({np.shape(value) for value in values}) == 1:
        return np.stack(values)
    stacked = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        stacked[i] = value
    return stacked
