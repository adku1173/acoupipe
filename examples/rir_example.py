import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

from acoupipe.config import HAVE_GPURIR
from acoupipe.datasets.utils import get_absorption_coeff

rng = np.random.RandomState(1)
rng2 = np.random.RandomState(1)

alpha = get_absorption_coeff(rng, realistic_walls=True)
alpha_off = get_absorption_coeff(rng2, realistic_walls=False)

print(alpha)
print(alpha_off)

if HAVE_GPURIR:
    from acoupipe.datasets.transfer import TransferGpuRIR

    mics = ac.MicGeom(mpos_tot=np.abs(np.random.normal(size=64*3)).reshape(3, -1))
    grid = ac.ImportGrid(gpos_file=np.abs(np.random.normal(size=6)).reshape(3, -1))
    trans = TransferGpuRIR(sample_freq=51200, block_size=1024,
        mics=mics, grid=grid, room_size=[6, 6, 6], alpha=alpha_off.mean(1), rel_tdir=np.array([0.5, 0.5, 0.5]))
    rir = trans.rir
    transfer = trans.transfer()
    mag_transfer = 20*np.log10(np.abs(transfer))

    plt.figure()
    plt.title("RIR")
    plt.plot(rir[0, 0, :])

    plt.figure()
    plt.title("Transfer")
    plt.plot(trans.fftfreq(), mag_transfer[0, 0, :])
    plt.xscale("log")






