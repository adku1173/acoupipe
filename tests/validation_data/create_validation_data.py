
from os import path
import numpy as np
from acoupipe import Dataset1, Dataset2
from acoular import MicGeom

mpos_tot = np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])

for Dataset in [Dataset1, Dataset2]:#,(Dataset2,config2)]:

    for feature in ['sourcemap','csmtriu','csm']:
        dataset = Dataset(split="training",
                            size=1,
                            startsample=20,
                            f = 1000,
                            features=[feature],
                            mics=MicGeom(mpos_tot=mpos_tot))
        dataset.grid.increment = 1/5*1.5
        for data in dataset.generate(tasks=1):
            np.save(path.join(".","validation_data",f"{type(dataset).__name__}_loc.npy"),data['loc'])
            np.save(path.join(".","validation_data",f"{type(dataset).__name__}_p2.npy"),data['p2'])
            np.save(path.join(".","validation_data",f"{type(dataset).__name__}_{feature}.npy"),data[feature])
