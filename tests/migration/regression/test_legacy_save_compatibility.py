# ruff: noqa: D100,D103,S101

from acoupipe.datasets.synthetic import DatasetSynthetic, DatasetSyntheticConfig

import h5py


def test_legacy_save_h5_still_writes_requested_features(tmp_path):
    path = tmp_path / 'legacy.h5'
    dataset = DatasetSynthetic(
        config=DatasetSyntheticConfig(signal_length=0.01, max_nsources=1, min_nsources=1),
    )

    dataset.save_h5(features=['loc'], size=1, name=str(path), split='training', progress_bar=False)

    with h5py.File(path, 'r') as h5:
        assert '0/loc' in h5
        assert '0/seeds' in h5
