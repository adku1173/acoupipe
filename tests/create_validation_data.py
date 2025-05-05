import argparse
import itertools

import numpy as np
from test_datasets import (
    IMPLEMENTED_FEATURES,
    TEST_SIGNAL_LENGTH,
    start_idx,
    validation_data_path,
)

from acoupipe.datasets.experimental import DatasetMIRACLE
from acoupipe.datasets.synthetic import DatasetSynthetic, DatasetSyntheticTestConfig


def create_synthetic_dataset(mode, **kwargs):
    """Creates a DatasetSynthetic instance."""
    config = DatasetSyntheticTestConfig(**kwargs)
    return DatasetSynthetic(config=config, mode=mode, **kwargs)

def create_miracle_dataset(mode, **kwargs):
    """Creates a DatasetMIRACLE instance."""
    return DatasetMIRACLE(mode=mode, **kwargs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, nargs="+", default=["DatasetSynthetic", "DatasetMIRACLE"])
    parser.add_argument("--features", type=str, nargs="+", default=IMPLEMENTED_FEATURES)
    parser.add_argument("--mode", type=str, nargs="+", default=["analytic", "welch", "wishart"])
    parser.add_argument("--frequencies", nargs="+", type=int, default=[None, 1000])
    parser.add_argument("--nums", nargs="+", type=int, default=[0, 3])
    args = parser.parse_args()

    print(f"Creating validation data for features {args.features} and modes {args.mode}")

    frequencies = list(args.frequencies)
    nums = list(args.nums)
    if "DatasetSynthetic" in args.dataset:
        for mode, f, num, feature in itertools.product(args.mode, frequencies, nums, args.features):
            if f is None and num != 0:
                continue
            if feature in ["time_data", "spectrogram"] and mode != "welch":
                continue
            dataset = create_synthetic_dataset(mode=mode)
            data = next(dataset.generate(
                f=f, num=num, features=[feature],
                split="training", progress_bar=False,
                size=1, start_idx=start_idx))
            np.save(
                validation_data_path / f"DatasetSynthetic_{feature}_f{f}_num{num}_mode{mode}.npy", data[feature])

    if "DatasetMIRACLE" in args.dataset:
        for mode, f, num, feature in itertools.product(args.mode, [1000], [0], args.features):
            if feature in ["time_data", "spectrogram"] and mode != "welch":
                continue
            dataset = create_miracle_dataset(mode=mode, signal_length=TEST_SIGNAL_LENGTH)
            data = next(dataset.generate(
                f=f, num=num, features=[feature],
                split="training", progress_bar=False,
                size=1, start_idx=start_idx))
            np.save(
                validation_data_path / f"DatasetMIRACLE_{feature}_f{f}_num{num}_mode{mode}.npy", data[feature])
