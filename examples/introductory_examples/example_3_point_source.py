"""
Sampling Point Source Scenarios
===============================

By combining several samplers, AcouPipe can generate randomized acoustic scenes.
This example sets up a microphone array observing a set of point sources driven by
white noise, and then randomizes three things at once: the source strengths (their
``rms`` values), the source positions, and which subset of sources is active. For
each random scene a beamforming map is computed.
"""

# %%
# First, the necessary Python modules and objects are imported. ``PowerSpectra``,
# ``SteeringVector`` and ``BeamformerBase`` come from Acoular, while the three
# samplers come from AcouPipe.

from pathlib import Path

import acoular as ac
from acoupipe.sampler import NumericAttributeSampler, PointSourceSampler, SourceSetSampler

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# %%
# Caching is disabled so that every re-sampled scene is recomputed from scratch.
# Three independent random states keep the three samplers reproducible and mutually
# independent.

ac.config.global_caching = 'none'

rng1 = np.random.RandomState(1)  
rng2 = np.random.RandomState(2)
rng3 = np.random.RandomState(3)

# %%
# The distance between the source plane and the microphone array is set to 0.5 m.

z = 0.5

# %%
# The source strengths are drawn from a Rayleigh distribution and the position
# offsets from a normal distribution.

rayleigh_dist = scipy.stats.rayleigh(scale=5.0)
normal_dist = scipy.stats.norm(loc=0, scale=0.1688)

# %%
# A 64 channel microphone array geometry shipped with Acoular is loaded, and ten
# point sources with individual white noise signals are placed at the origin of the
# source plane.

mg = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')

# create white noise signals and pointsources
wn_list = []
ps_list = []
for i in range(10):
    wn_list.append(ac.WNoiseGenerator(sample_freq=51200, seed=10 + i, rms=1.0, num_samples=51200))
    ps_list.append(ac.PointSource(signal=wn_list[i], mics=mg, loc=(0.0, 0.0, z)))

# %%
# Next, the Acoular processing chain is set up. A ``SourceMixer`` combines the
# active sources into a single signal. From this signal, a ``PowerSpectra`` object
# computes the cross spectral matrix. Finally, a ``BeamformerBase`` maps the cross
# spectral matrix onto a ``RectGrid`` that spans the source plane.

sm = ac.SourceMixer(sources=ps_list)
ps = ac.PowerSpectra(source=sm, block_size=512, window='Hanning')
rg = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=z, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg, steer_type='true location')
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# Three samplers are defined. The first randomizes the ``rms`` of every noise
# generator, the second shifts each source in the x-y plane within the given
# bounds, and the third selects three distinct active sources for the mixer.

rms_sampling = NumericAttributeSampler(
    random_var=rayleigh_dist,
    target=wn_list,
    attribute='rms',
    random_state=rng1,
)

ps_sampling = PointSourceSampler(
    random_var=normal_dist,
    target=ps_list,
    ldir=np.array([[1.0], [1.0], [0.0]]),
    x_bounds=(-0.5, 0.5),
    y_bounds=(-0.5, 0.5),
    random_state=rng2,
)

src_sampling = SourceSetSampler(
    target=[sm],
    set=ps_list,
    nsources=3,
    replace=False,
    random_state=rng3,
)

# %%
# For a few random scenes we call ``sample()`` on each sampler, recompute the
# beamforming map at 2000 Hz, and plot it together with the true source positions
# (red crosses).

cfreq = 2000
for case in range(3):
    rms_sampling.sample()
    ps_sampling.sample()
    src_sampling.sample()

    Lm = ac.L_p(bb.synthetic(cfreq, 1))

    fig, ax = plt.subplots()
    im = ax.imshow(Lm.T, origin='lower', vmin=Lm.max() - 15, extent=rg.extent, interpolation='bicubic')
    for src in sm.sources:
        x, y, _ = src.loc
        ax.plot(x, y, marker='x', color='red')
    ax.set_title(f'random scene {case + 1}')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    fig.colorbar(im, ax=ax, label='SPL / dB')
    plt.show()