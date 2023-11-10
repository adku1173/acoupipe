

import numpy as np
from test_datasets import IMPLEMENTED_FEATURES, TestDatasetSynthetic1, start_idx, validation_data_path

if __name__ == "__main__":

    for mode in ["analytic","welch","wishart"]:
        for f in [None,1000]:
            for num in [0,3]:
                if f is None and num != 0:
                    continue
                for feature in IMPLEMENTED_FEATURES:
                    if feature in ["time_data","spectrogram"] and mode != "welch":
                        continue
                    dataset = TestDatasetSynthetic1.create_dataset(
                        mode=mode,)
                    data = next(dataset.generate(
                        f=f,num=num,features=[feature],
                        split="training",progress_bar=False,
                        size=1,start_idx=start_idx))
                    np.save(
                        validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy",data[feature])
