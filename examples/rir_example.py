from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.transfer import TransferGpuRIR
from acoupipe.datasets.utils import get_absorption_coeff

rng = np.random.RandomState(1)
rng2 = np.random.RandomState(1)


block_size = 128
fs = 13720
signal_length = 10  # seconds
alpha_off = get_absorption_coeff(rng2, realistic_walls=False)
# alpha_off = get_absorption_coeff(rng2, realistic_walls=True)
# alpha_off = np.ones_like(alpha_off)

# source locations: array of (3,num_sources):
pos = np.array([[-0.5602875, -0.43781025], [0.92770928, -0.03813699], [1.5, 1.5]])
num_sources = pos.shape[1]

# source strengths (all equal)
Q = np.zeros((block_size // 2 + 1, num_sources, num_sources), dtype=np.complex128)
for i in range(num_sources):
    Q[:, i, i] = 1

# ref position
ref = np.array([-0.021, -0.063, 0.0])

# room parameters
room_size = np.array([2.18040357, 2.85165386, 2.17179724])
alpha = np.array([0.14632144, 0.32326808, 0.67843576, 0.81814866, 0.49110407, 0.86437483])  # absorption coefficients per wall
origin = np.array([1.08414871, 1.40887475, 0.18653786])  # origin of the room


mics = ac.MicGeom(file=Path(ac.__file__).parent / "xml" / "minidsp_uma-16_mirrored.xml")
source_grid = ac.ImportGrid(pos=pos)

# define the transfer function (via ISM model)
trans = TransferGpuRIR(
    ref=ref,
    sample_freq=fs,
    block_size=block_size,
    mics=mics,
    grid=source_grid,
    room_size=room_size,
    alpha=alpha_off[:, 0],
    origin=origin,
)

csm = PowerSpectraAnalytic(
    Q=Q,
    transfer=trans,
    mode="wishart",
    block_size=block_size,
    overlap="50%",
    numsamples=signal_length * fs,
    sample_freq=fs,
)


# get the CSM for the Transformer:
csm_full = csm.csm
print(csm_full.shape)

# Beamforming
rg = ac.RectGrid(
    x_min=-1.5,
    x_max=1.5,
    y_min=-1.5,
    y_max=1.5,
    z=1.5,
    increment=0.05,
)

steer = ac.SteeringVector(
    mics=mics,
    grid=rg,
    ref=trans.ref,
)

bb = ac.BeamformerBase(
    freq_data=csm,
    steer=steer,
)

Lm = ac.L_p(bb.synthetic(4000, 3)).T

# source map
fig = plt.figure()
plt.title("Beamforming Map")
extent = rg.extend()
plt.imshow(Lm, extent=extent, origin="lower", vmax=Lm.max(), vmin=Lm.max() - 20)
plt.colorbar(label="Sound Pressure Level (dB)")
# annotate source locations
for loc in pos.T:
    plt.scatter(loc[0], loc[1])

# plot transfer function

plt.figure()
plt.title("Transfer Function Power Spectra")


# plot ref channel
for i in range(0, num_sources):
    tf = trans.transfer()[i]
    tfps = np.real(tf * tf.conj())

    plt.plot(trans.fftfreq(), 10 * np.log10(tfps[0]), label="ref")

    for m in range(1, 15):
        plt.plot(trans.fftfreq(), 10 * np.log10(tfps[m]), label=f"mic {m + 1}")
plt.legend()
plt.ylim(-60, 60)
plt.xscale("log")
plt.xlim(500, 8000)
