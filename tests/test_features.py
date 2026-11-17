"""Tests for the feature handling of the datasets.

Covers the assembly of the feature functions and the cleanup of the generated samples.
"""

import inspect

from acoupipe.datasets.features import create_feature
from acoupipe.datasets.synthetic import DatasetSynthetic

from tests.synthetic_test_config import DatasetSyntheticTestConfig

import numpy as np


def test_cleanup(create_dataset):
    """Test the cleanup of DatasetSynthetic.

    Checks that auxiliary keys of the prepare function are removed from the samples
    and that the outputs of custom features are kept.
    """

    def custom_with_data(sampler, data):  # noqa: ARG001
        return {'custom_with_data': 1}

    def custom_without_data(sampler):  # noqa: ARG001
        return {'custom_without_data': 1}

    features = [
        'loc',
        create_feature(custom_with_data, name='custom_with_data', shape=(), dtype=np.int64),
        create_feature(custom_without_data, name='custom_without_data', shape=(), dtype=np.int64),
    ]
    dataset = create_dataset(mode='analytic')
    data = next(dataset.generate(features=features, f=1000, num=0, split='training', size=1, progress_bar=False))

    assert set(data.keys()) == {'idx', 'seeds', 'loc', 'custom_with_data', 'custom_without_data'}


def test_feature_order():
    """Test the order of calculation: prepare function, default features, custom features.

    The prepare function and the default features of the config are wrapped to record
    their calls. The custom feature is listed before the default feature on purpose,
    since the order of calculation does not depend on the order of the requested features.
    """
    call_order = []

    def record(name, func):
        def wrapped(sampler, data):
            call_order.append(name)
            if 'data' in inspect.signature(func).parameters:
                return func(sampler=sampler, data=data)
            return func(sampler=sampler)

        return wrapped

    class OrderTestConfig(DatasetSyntheticTestConfig):
        def get_prepare_func(self):
            return record('prepare', super().get_prepare_func())

        def get_default_features(self, features, f, num):
            return [
                create_feature(record(feat.name, feat.get_feature_func()), feat.name, feat.shape, feat.dtype)
                for feat in super().get_default_features(features, f, num)
            ]

    def custom(sampler, data):  # noqa: ARG001
        call_order.append('custom')
        return {'custom': 1}

    features = [create_feature(custom, name='custom', shape=(), dtype=np.int64), 'csm']
    dataset = DatasetSynthetic(config=OrderTestConfig(mode='analytic'), mode='analytic')
    next(dataset.generate(features=features, f=1000, num=0, split='training', size=1, progress_bar=False))

    assert call_order == ['prepare', 'csm', 'custom']
