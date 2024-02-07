import argparse

import numpy as np
from test_datasets import (
    IMPLEMENTED_FEATURES,
    TEST_SIGNAL_LENGTH,
    TestDatasetSynthetic,
    TestMIRACLEDataset,
    start_idx,
    validation_data_path,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, nargs="+", default=IMPLEMENTED_FEATURES)
    parser.add_argument("--mode", type=str, nargs="+", default=["analytic","welch","wishart"])
    args = parser.parse_args()

    print(f"Creating validation data for features {args.features} and modes {args.mode}")

    for mode in args.mode:
        dataset = TestDatasetSynthetic.create_dataset(
            mode=mode,)
        for f in [None,1000]:
            for num in [0,3]:
                if f is None and num != 0:
                    continue
                for feature in args.features:
                    if feature in ["time_data","spectrogram"] and mode != "welch":
                        continue
                    # Dataset Synthetic
                    data = next(dataset.generate(
                        f=f,num=num,features=[feature],
                        split="training",progress_bar=False,
                        size=1,start_idx=start_idx))
                    np.save(
                        validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy",data[feature])

    for mode in args.mode:
        dataset = TestMIRACLEDataset.create_dataset(
            mode=mode,signal_length=TEST_SIGNAL_LENGTH)
        for f in [1000]:
            for num in [0]:
                for feature in args.features:
                    if feature in ["time_data","spectrogram"] and mode != "welch":
                        continue
                    data = next(dataset.generate(
                        f=f,num=num,features=[feature],
                        split="training",progress_bar=False,
                        size=1,start_idx=start_idx))
                    np.save(
                        validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy",data[feature])
