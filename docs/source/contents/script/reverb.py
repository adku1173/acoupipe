
#%%
import acoular as ac
import matplotlib.pyplot as plt
import pyfar

from acoupipe.datasets.synthetic import DatasetSynthetic

mode = "wishart"

#%%

dataset = DatasetSyntheticISM(mode=mode, fs=51200,
mic_pos_noise=False, max_nsources=1, mic_sig_noise=False,
            #signal_length=20
            )
dataset.config.fft_params["block_size"] = 4096

# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=[
    "sourcemap",
    "csm","loc", "f", "source_strength_estimated"
    ],
    split="training", size=10, f=[2000], num=0)
data_sample = next(data_generator)

print("sourcemap:", data_sample["sourcemap"].sum())
print("max sourcemap level:", ac.L_p(data_sample["sourcemap"]).max())

if data_sample.get("csm") is not None:
    print(ac.L_p(data_sample["csm"][:,63,63]))

if data_sample.get("source_strength_estimated") is not None:
    print(ac.L_p(data_sample["source_strength_estimated"]))

extent = dataset.config.grid.extend()

# sound pressure level
Lm = ac.L_p(data_sample["sourcemap"]).T
Lm_max = Lm.max()
Lm_min = Lm.max() - 20

# plot sourcemap
fig = plt.figure()
plt.title("Beamforming Map)")
plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")
plt.colorbar(label="Sound Pressure Level (dB)")
# plot source locations
# for loc in data_sample["loc"].T:
#     plt.scatter(loc[0], loc[1])
plt.xlabel("x (m)")
plt.ylabel("y (m)")


if dataset.config.sources[0].kernel.size > 0:

    plt.figure()
    kernels = dataset.config.sources[0].kernel
    for kernel in kernels.T:
        plt.plot(kernel)
        break

    # plot fft spectrum
    plt.figure()
    signal = pyfar.Signal(kernels.T[:2], sampling_rate=dataset.config.fs)
    pyfar.plot.freq(signal)
    #pyfar.plot.phase(signal)



#%%

dataset = DatasetSynthetic(mode=mode, max_nsources=1)

# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=["sourcemap","loc", "f", "source_strength_estimated"],
                                    split="training", size=10, f=[2000], num=3)

data_sample = next(data_generator)
print(data_sample["sourcemap"].sum())
print(ac.L_p(data_sample["source_strength_estimated"]))


extent = dataset.config.grid.extend()

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


from acoupipe.datasets.experimental import DatasetMIRACLE

dataset = DatasetMIRACLE(mode=mode, max_nsources=1)

# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(features=["sourcemap","loc", "f", "source_strength_estimated"],
                                    split="training", size=10, f=[2000], num=3)

data_sample = next(data_generator)

print(ac.L_p(data_sample["source_strength_estimated"]))

extent = dataset.config.grid.extend()

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

plt.figure()
kernels = dataset.config.sources[0].kernel
for kernel in kernels.T:
    plt.plot(kernel)
    break




#%%
from pathlib import Path

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "ISM_sourcemap.png", dpi=300)



# %%
