
#%%
from acoupipe.datasets.synthetic import DatasetSynthetic

dataset = DatasetSynthetic()
# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=["sourcemap","loc", "f"],
                                    split="training", size=10, f=[2000], num=0)
data_sample = next(data_generator)



#%%

import acoular as ac
import matplotlib.pyplot as plt

extent = dataset.config.msm_setup.grid.extend()

# sound pressure level
Lm = ac.L_p(data_sample["sourcemap"]).T
Lm_max = Lm.max()
Lm_min = Lm.max() - 20

# plot sourcemap
fig = plt.figure()
plt.title(f'Beamforming Map (f={data_sample["f"][0]} Hz)')
plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")
plt.colorbar(label="Sound Pressure Level (dB)")
# plot source locations
for loc in data_sample["loc"].T:
    plt.scatter(loc[0], loc[1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")
#%%
from pathlib import Path

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "quickstart_sourcemap.png", dpi=300)



# %%
