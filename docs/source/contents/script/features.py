#%%

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

from acoupipe.datasets.synthetic import DatasetSynthetic

#%% targetmap example

fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
fig.suptitle("Target Sourcemap ($f=2000$ Hz, $J=5$)", fontsize=12)

mode = "analytic"
for i, feature in enumerate(["targetmap_analytic", "targetmap_estimated"]):

    dataset = DatasetSynthetic(mode=mode)
    # generate data for frequency 2000 Hz (single frequency)
    data_generator = dataset.generate(features=[feature, "f"],
                                        split="training", size=1, f=[2000], num=0, start_idx=1)
    data_sample = next(data_generator)

    extent = dataset.config.grid.extend()

    # sound pressure level
    Lm = ac.L_p(data_sample[feature]).T
    Lm_max = Lm.max()
    Lm_min = Lm.max() - 20

    axs[i].set_title(f"{feature}")
    axs[i].imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "targetmap_example.png", dpi=300)


#%% sourcemap_example

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
fig.suptitle("Sourcemap ($f=2000$ Hz, $J=5$)", fontsize=12)


for i, mode in enumerate(["welch", "analytic", "wishart"]):

    dataset = DatasetSynthetic(mode=mode)
    # generate data for frequency 2000 Hz (single frequency)
    data_generator = dataset.generate(features=["sourcemap","loc", "f"],
                                        split="training", size=1, f=[2000], num=0, start_idx=1)
    data_sample = next(data_generator)

    extent = dataset.config.grid.extend()

    # sound pressure level
    Lm = ac.L_p(data_sample["sourcemap"]).T
    Lm_max = Lm.max()
    Lm_min = Lm.max() - 20

    axs[i].set_title(f"mode={mode}")
    axs[i].imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower",interpolation="bicubic")

    # # plot source locations
    # for loc in data_sample["loc"].T:
    #     axs[i].scatter(loc[0], loc[1])
    # axs[i].set_xlabel("x (m)")
    # axs[i].set_ylabel("y (m)")


dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "sourcemap_example.png", dpi=300)

#%% csm example

fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
fig.suptitle("CSM ($f=2000$ Hz, $J=5$)", fontsize=12)


for i, mode in enumerate(["welch", "analytic", "wishart"]):
    for j in range(2):

        dataset = DatasetSynthetic(mode=mode)
        # generate data for frequency 2000 Hz (single frequency)
        data_generator = dataset.generate(features=["csm","loc", "f"],
                                            split="training", size=1, f=[2000], num=0, start_idx=1)
        data_sample = next(data_generator)

        extent = dataset.config.grid.extend()

        # sound pressure level
        if j == 0:
            csm = np.real(data_sample["csm"][0])
            if i == 0:
                axs[j,i].set_ylabel("Re")
        else:
            csm = np.imag(data_sample["csm"][0])
            if i == 0:
                axs[j,i].set_ylabel("Im")
        if j == 0:
            axs[j,i].set_title(f"mode={mode}")

        axs[j,i].imshow(csm)
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "csm_example.png", dpi=300)


#%% csmtriu example

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
fig.suptitle("compressed CSM ($f=2000$ Hz, $J=5$)", fontsize=12)


for i, mode in enumerate(["welch", "analytic", "wishart"]):

    dataset = DatasetSynthetic(mode=mode)
    # generate data for frequency 2000 Hz (single frequency)
    data_generator = dataset.generate(features=["csmtriu","loc", "f"],
                                        split="training", size=1, f=[2000], num=0, start_idx=1)
    data_sample = next(data_generator)

    extent = dataset.config.grid.extend()

    # sound pressure level
    csmtriu = np.real(data_sample["csmtriu"][0])
    axs[i].set_title(f"mode={mode}")
    axs[i].imshow(csmtriu)
    axs[i].set_xticks([])
    axs[i].set_yticks([])

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "csmtriu_example.png", dpi=300)


#%% eigmode example

fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True, sharex=True)
fig.suptitle("CSM Eigenvalues ($f=2000$ Hz, $J=5$)", fontsize=12)

for mode in ["welch", "analytic", "wishart"]:

    dataset = DatasetSynthetic(mode=mode)
    # generate data for frequency 2000 Hz (single frequency)
    data_generator = dataset.generate(features=["eigmode","loc", "f"],
                                        split="training", size=1, f=[2000], num=0, start_idx=1)
    data_sample = next(data_generator)

    extent = dataset.config.grid.extend()

    # sound pressure level
    eigmode = data_sample["eigmode"][0]
    eigval = 10*np.log10(np.linalg.norm(eigmode,axis=0)[::-1])
    axs.plot(eigval, label=f"{mode}")
    axs.set_xlabel("Eigenvalue index")
    axs.set_ylabel("Eigenvalue (dB)")
plt.legend()
plt.tight_layout()

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "eigval_example.png", dpi=300)

# %% analytic example


fig, axs = plt.subplots(1, 1, figsize=(4, 4), sharey=True, sharex=True)
fig.suptitle("Analytic Source Strength ($J=5$, idx=1)", fontsize=12)

dataset = DatasetSynthetic(mode="analytic")
# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=["source_strength_analytic","f"],
                                    split="training", size=1, num=0, start_idx=1)
data_sample = next(data_generator)

for j, source_strength in enumerate(data_sample["source_strength_analytic"].T):
    axs.plot(data_sample["f"],ac.L_p(source_strength), label=f"source {j+1}")
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Sound Pressure Level (dB)")
plt.semilogx()
plt.legend()
plt.tight_layout()

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "source_strength_analytic_example.png", dpi=300)


# %% estimated

fig, axs = plt.subplots(1, 1, figsize=(4, 4), sharey=True, sharex=True)
fig.suptitle("Estimated Source Strength ($J=5$, idx=1)", fontsize=12)

dataset = DatasetSynthetic(mode="wishart")
# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=["source_strength_estimated","f"],
                                    split="training", size=1, num=0, start_idx=1)
data_sample = next(data_generator)

for j, source_strength in enumerate(data_sample["source_strength_estimated"].T):
    axs.plot(data_sample["f"],ac.L_p(source_strength), label=f"source {j+1}")
axs.set_xlabel("Frequency (Hz)")
axs.set_ylabel("Sound Pressure Level (dB)")
plt.semilogx()
plt.legend()
plt.tight_layout()

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "source_strength_estimated_example.png", dpi=300)

# %%
