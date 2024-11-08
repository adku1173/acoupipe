
#%%


from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
from modelsdfg.transformer.datasets import ConfigUMAReverb

from acoupipe.datasets.reverb import DatasetISM

#%%


mics = ac.MicGeom(from_file=Path(ac.__file__).parent / "xml" / "minidsp_uma-16_mirrored.xml")

config = ConfigUMAReverb(
        mode="wishart",
        random_signal_length = True,
        mic_pos_noise=False,
        )

dataset = DatasetISM(
    # mode = "analytic",
    # max_nsources = 1,
    # mic_sig_noise = False,
    # random_signal_length = False,
    # mic_pos_noise = False,
    # signal_length = 10,
    config=config,
    tasks=1, num_gpus=1,
    )
# generate data for frequency 2000 Hz (single frequency)
data_generator = dataset.generate(
    features=["sourcemap", "csm", "source_strength_analytic", "source_strength_estimated", "loc", "f"],
    #features=["csm"],
                                    split="training", size=1000, f=[4000], num=0, start_idx=1)
data_sample = next(data_generator)

# for data_sample in enumerate(data_generator):
#     pass

if data_sample.get("source_strength_analytic") is not None:
    print(ac.L_p(data_sample["source_strength_analytic"].sum()))
if data_sample.get("source_strength_estimated") is not None:
    print(ac.L_p(data_sample["source_strength_estimated"]))
if data_sample.get("csm") is not None:
    print(ac.L_p(data_sample["csm"][0,0,0]))
if data_sample.get("sourcemap") is not None:
    print(ac.L_p(data_sample["sourcemap"]).max())

if data_sample.get("sourcemap") is not None and data_sample.get("loc") is not None and data_sample.get("f") is not None:

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



import numpy as np

plt.figure()
freqs = config.msm_setup.freq_data.transfer.fftfreq()
H_j0 = config.msm_setup.freq_data.transfer._get_r0_transfer().squeeze()
H_rm_0 = config.msm_setup.freq_data.transfer._get_rm_transfer()[:,0].squeeze()

plt.plot(freqs,10*np.log10(np.abs(H_j0)))
plt.plot(freqs,10*np.log10(np.abs(H_rm_0)))
plt.show()


# #%%
# # generate data for frequency 2000 Hz (single frequency)
# data_generator = dataset.generate(features=[
#     "sourcemap",
#     "csm","loc", "f", "source_strength_estimated"
#     ],
#     split="training", size=10, f=[2000], num=0)
# data_sample = next(data_generator)

# print("sourcemap:", data_sample["sourcemap"].sum())
# print("max sourcemap level:", ac.L_p(data_sample["sourcemap"]).max())

# if data_sample.get("csm") is not None:
#     print(ac.L_p(data_sample["csm"][:,63,63]))

# if data_sample.get("source_strength_estimated") is not None:
#     print(ac.L_p(data_sample["source_strength_estimated"]))

# extent = dataset.config.grid.extend()

# # sound pressure level
# Lm = ac.L_p(data_sample["sourcemap"]).T
# Lm_max = Lm.max()
# Lm_min = Lm.max() - 20

# # plot sourcemap
# fig = plt.figure()
# plt.title("Beamforming Map)")
# plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")
# plt.colorbar(label="Sound Pressure Level (dB)")
# # plot source locations
# # for loc in data_sample["loc"].T:
# #     plt.scatter(loc[0], loc[1])
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")


# if dataset.config.sources[0].kernel.size > 0:

#     plt.figure()
#     kernels = dataset.config.sources[0].kernel
#     for kernel in kernels.T:
#         plt.plot(kernel)
#         break

#     # plot fft spectrum
#     plt.figure()
#     signal = pyfar.Signal(kernels.T[:2], sampling_rate=dataset.config.fs)
#     pyfar.plot.freq(signal)
#     #pyfar.plot.phase(signal)



# #%%

# dataset = DatasetSynthetic(mode=mode, max_nsources=1)

# # generate data for frequency 2000 Hz (single frequency)
# data_generator = dataset.generate(features=["sourcemap","loc", "f", "source_strength_estimated"],
#                                     split="training", size=10, f=[2000], num=3)

# data_sample = next(data_generator)
# print(data_sample["sourcemap"].sum())
# print(ac.L_p(data_sample["source_strength_estimated"]))


# extent = dataset.config.grid.extend()

# # sound pressure level
# Lm = ac.L_p(data_sample["sourcemap"]).T
# Lm_max = Lm.max()
# Lm_min = Lm.max() - 20

# # plot sourcemap
# fig = plt.figure()
# plt.title(f'Beamforming Map (f={data_sample["f"][0]} Hz)')
# plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")
# plt.colorbar(label="Sound Pressure Level (dB)")
# # plot source locations
# for loc in data_sample["loc"].T:
#     plt.scatter(loc[0], loc[1])
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")

# #%%


# from acoupipe.datasets.experimental import DatasetMIRACLE

# dataset = DatasetMIRACLE(mode=mode, max_nsources=1)

# # generate data for frequency 2000 Hz (single frequency)
# data_generator = dataset.generate(features=["sourcemap","loc", "f", "source_strength_estimated"],
#                                     split="training", size=10, f=[2000], num=3)

# data_sample = next(data_generator)

# print(ac.L_p(data_sample["source_strength_estimated"]))

# extent = dataset.config.grid.extend()

# # sound pressure level
# Lm = ac.L_p(data_sample["sourcemap"]).T
# Lm_max = Lm.max()
# Lm_min = Lm.max() - 20

# # plot sourcemap
# fig = plt.figure()
# plt.title(f'Beamforming Map (f={data_sample["f"][0]} Hz)')
# plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower")
# plt.colorbar(label="Sound Pressure Level (dB)")
# # plot source locations
# for loc in data_sample["loc"].T:
#     plt.scatter(loc[0], loc[1])
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")

# plt.figure()
# kernels = dataset.config.sources[0].kernel
# for kernel in kernels.T:
#     plt.plot(kernel)
#     break




# #%%
# from pathlib import Path

# dpath = Path(__file__).parent.parent.parent / "_static"
# fig.savefig(dpath / "ISM_sourcemap.png", dpi=300)



# # %%
