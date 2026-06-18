# Copyright (c) 2026 AcouPipe Developers.
# Distributed under the terms of the BSD-3-Clause license.

import numpy as np
from traits.api import Any

from acoupipe.datasets import Config, ParameterSet
from acoupipe.datasets.features import BaseFeatureCatalog, create_feature


class RecordingFeature(BaseFeatureCatalog):
    """Feature object used to verify prepare/calculate lifecycle order."""

    record = Any()

    def get_prepare_func(self):
        def prepare(parameters, data):
            self.record.append(f'prepare:{self.name}:{parameters.marker}:{sorted(data)}')

        return prepare

    def get_feature_func(self):
        def calculate(parameters, data):
            self.record.append(f'calculate:{self.name}:{parameters.marker}:{sorted(data)}')
            return {self.name: np.int32(len(data) + 1)}

        return calculate


def test_feature_prepare_runs_immediately_before_own_calculation():
    record = []
    config = Config(parameters=ParameterSet.from_dict({'marker': 'sample'}))
    feature_a = RecordingFeature(name='a', dtype=np.int32, shape=(), record=record)
    feature_b = RecordingFeature(name='b', dtype=np.int32, shape=(), record=record)

    feature_collection = config.get_feature_collection([feature_a, feature_b], f=None, num=0)
    result = feature_collection.get_feature_funcs()(sampler={})

    assert result == {'a': np.int32(1), 'b': np.int32(2)}
    assert record == [
        'prepare:a:sample:[]',
        'calculate:a:sample:[]',
        "prepare:b:sample:['a']",
        "calculate:b:sample:['a']",
    ]


def test_feature_without_prepare_hook_still_works():
    feature = create_feature(
        feature_func=lambda _: {'plain': np.float32(1.0)},
        name='plain',
        dtype=np.float32,
        shape=(),
    )
    config = Config(parameters=ParameterSet())

    feature_collection = config.get_feature_collection([feature], f=None, num=0)
    result = feature_collection.get_feature_funcs()(sampler={})

    assert result == {'plain': np.float32(1.0)}


def test_registered_feature_callback_can_receive_parameters_and_data():
    config = Config(parameters=ParameterSet.from_dict({'scale': np.float32(2.0)}))

    config.feature('base', lambda parameters: np.float32(parameters.scale), dtype=np.float32, shape=())

    def calculate_scaled(parameters, data):
        return np.float32(data['base'] * parameters.scale)

    config.feature('scaled', calculate_scaled, dtype=np.float32, shape=())

    feature_collection = config.get_feature_collection(['base', 'scaled'], f=None, num=0)
    result = feature_collection.get_feature_funcs()(sampler={})

    assert result['base'] == np.float32(2.0)
    assert result['scaled'] == np.float32(4.0)


def test_legacy_config_prepare_hook_does_not_receive_parameters_by_signature_only():
    record = []

    class LegacyHookConfig(Config):
        def get_prepare_func(self):
            def prepare_hook(sampler, parameters=None):  # noqa: ARG001
                record.append(parameters)
                return {}

            return prepare_hook

    config = LegacyHookConfig(parameters=ParameterSet.from_dict({'marker': 'sample'}))
    config.feature('value', lambda: np.int32(1), dtype=np.int32, shape=())

    feature_collection = config.get_feature_collection(['value'], f=None, num=0)
    result = feature_collection.get_feature_funcs()(sampler={})

    assert result == {'value': np.int32(1)}
    assert record == [None]
