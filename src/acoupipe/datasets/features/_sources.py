"""Helpers to find the sources within a graph of Acoular objects."""

import acoular as ac


def get_point_sources_recursively(source):
    """Recursively get all point sources from a `acoular.TimeInOut` object.

    Parameters
    ----------
    source : instance of class `acoular.TimeInOut`
        the source object

    Returns
    -------
    list
        list of all sources
    """
    sources = []
    if isinstance(source, ac.PointSource):
        return [source]
    if isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_point_sources_recursively(source.source)
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return []
    return sources


def get_uncorrelated_noise_source_recursively(source):
    """Recursively get all uncorrelated noise sources from a `acoular.TimeInOut` object.

    Parameters
    ----------
    source : instance of class `acoular.TimeInOut`
        the source object

    Returns
    -------
    list
        list of all uncorrelated noise sources
    """
    sources = []
    if isinstance(source, ac.PointSource):
        return []
    if isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_uncorrelated_noise_source_recursively(source.source)
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return [source]
    return sources
