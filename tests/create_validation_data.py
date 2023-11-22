

import numpy as np
from test_datasets import IMPLEMENTED_FEATURES, TestDatasetSynthetic1, TestMIRACLEDataset, start_idx, validation_data_path

if __name__ == "__main__":

    signal_length = 0.5

    for mode in ["analytic","welch","wishart"]:
        dataset = TestDatasetSynthetic1.create_dataset(
            mode=mode,)
        for f in [None,1000]:
            for num in [0,3]:
                if f is None and num != 0:
                    continue
                for feature in IMPLEMENTED_FEATURES:
                    if feature in ["time_data","spectrogram"] and mode != "welch":
                        continue
                    # Dataset Synthetic
                    data = next(dataset.generate(
                        f=f,num=num,features=[feature],
                        split="training",progress_bar=False,
                        size=1,start_idx=start_idx))
                    np.save(
                        validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy",data[feature])

    for mode in ["analytic","welch","wishart"]:
        dataset = TestMIRACLEDataset.create_dataset(
            mode=mode,signal_length=signal_length)
        for f in [1000]:
            for num in [0]:
                for feature in IMPLEMENTED_FEATURES:
                    if feature in ["time_data","spectrogram"] and mode != "welch":
                        continue
                    data = next(dataset.generate(
                        f=f,num=num,features=[feature],
                        split="training",progress_bar=False,
                        size=1,start_idx=start_idx))
                    np.save(
                        validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy",data[feature])
