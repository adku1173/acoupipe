"""Tests for the feature handling of the datasets.

Covers the assembly of the feature functions and the cleanup of the generated samples.
"""

from acoupipe.datasets.features import create_feature

import numpy as np


def test_cleanup(create_dataset):
    """Test the cleanup of DatasetSynthetic.

    Checks that auxiliary keys of the prepare function are removed from the samples.
    Also pins the bug that the cleanup removes the output of custom features.
    """

    def custom_with_data(sampler, data):  # noqa ARG001
        return {'custom_with_data': 1}

    def custom_without_data(sampler):  # noqa ARG001
        return {'custom_without_data': 1}

    features = [
        'loc',
        create_feature(custom_with_data, name='custom_with_data', shape=(), dtype=np.int64),
        create_feature(custom_without_data, name='custom_without_data', shape=(), dtype=np.int64),
    ]
    dataset = create_dataset(mode='analytic')
    data = next(dataset.generate(features=features, f=1000, num=0, split='training', size=1, progress_bar=False))

    assert set(data.keys()) == {'idx', 'seeds', 'loc'}
