"""Feature catalog package for dataset generation internals."""

import warnings

from acoupipe.datasets.features.catalog import (
    AnalyticNoiseStrengthFeature,
    AnalyticSourceStrengthFeature,
    CSMFeature,
    CSMtriuFeature,
    EigmodeFeature,
    EstimatedNoiseStrengthFeature,
    EstimatedSourceStrengthFeature,
    Feature,
    LocFeature,
    SourcemapFeature,
    SpectraFeature,
    SpectrogramFeature,
    TargetmapFeature,
    TimeDataFeature,
    combine_feature_funcs,
    create_feature,
    get_tf_mappers,
    remove_unrequested,
)

__all__ = [
    'AnalyticNoiseStrengthFeature',
    'AnalyticSourceStrengthFeature',
    'CSMFeature',
    'CSMtriuFeature',
    'EigmodeFeature',
    'EstimatedNoiseStrengthFeature',
    'EstimatedSourceStrengthFeature',
    'Feature',
    'LocFeature',
    'SourcemapFeature',
    'SpectraFeature',
    'SpectrogramFeature',
    'TargetmapFeature',
    'TimeDataFeature',
    'combine_feature_funcs',
    'create_feature',
    'get_tf_mappers',
    'remove_unrequested',
]


def __getattr__(name):
    """Return :class:`Feature` for the deprecated name ``BaseFeatureCatalog``."""
    if name == 'BaseFeatureCatalog':
        warnings.warn(
            'BaseFeatureCatalog is deprecated, use Feature instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return Feature
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
