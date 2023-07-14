
from os import path

import numpy as np

from acoupipe.datasets.dataset1 import DEFAULT_GRID as DEFAULT_GRID1
from acoupipe.datasets.dataset1 import DEFAULT_MICS as DEFAULT_MICS1
from acoupipe.datasets.dataset1 import Dataset1
from acoupipe.datasets.dataset2 import DEFAULT_GRID as DEFAULT_GRID2
from acoupipe.datasets.dataset2 import DEFAULT_MICS as DEFAULT_MICS2
from acoupipe.datasets.dataset2 import Dataset2

dirpath = path.dirname(path.abspath(__file__))

DEFAULT_MICS1.mpos_tot = np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])
DEFAULT_MICS2.mpos_tot = DEFAULT_MICS1.mpos_tot.copy()

DEFAULT_GRID1.increment = 1/5*1.5
DEFAULT_GRID2.increment = 1/5*1.5

i = 1
for feature in ["sourcemap","csmtriu","csm", "eigmode"]:
    dataset = Dataset1(
                        f = 1000,
                        features=[feature])
    data = next(dataset.generate(split="training",tasks=1, progress_bar=False, size=1))
    if i == 1:
        np.save(path.join(f"{type(dataset).__name__}_loc.npy"),data["loc"])
        np.save(path.join(f"{type(dataset).__name__}_p2.npy"),data["p2"])
    np.save(path.join(f"{type(dataset).__name__}_{feature}.npy"),data[feature])
    i += 1

i = 1
dataset = Dataset2(
            f = 1000,
            features=["sourcemap","csmtriu","csm", "eigmode"])

for sample_noise in [False, True]:
    for sample_wishart in [False, True]:
        for sample_spectra in [False, True]:
            dataset.sample_wishart=sample_wishart
            dataset.sample_noise=sample_noise
            dataset.sample_spectra=sample_spectra
            data = next(dataset.generate(split="training", tasks=1, progress_bar=False,size=1,))
            if i == 1:
                np.save(path.join(f"{type(dataset).__name__}_loc.npy"),data["loc"])
            for feature in ["sourcemap","csmtriu","csm", "eigmode"]:
                np.save(path.join(f"{type(dataset).__name__}_p2_wishart{sample_wishart}_noise{sample_noise}_spectra{sample_spectra}.npy"),data["p2"])
                np.save(path.join(f"{type(dataset).__name__}_{feature}_wishart{sample_wishart}_noise{sample_noise}_spectra{sample_spectra}.npy"),data[feature])
            i += 1
