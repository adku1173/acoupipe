from pathlib import Path

import acoular as ac
import numpy as np
import scipy.stats
from pylab import colorbar, figure, imshow, plot, show

from acoupipe.sampler import NumericAttributeSampler, PointSourceSampler, SourceSetSampler

ac.config.global_caching = 'none'

rng1 = np.random.RandomState(1)  # scipy listens to numpy random seed (when scipy seed is None)
rng2 = np.random.RandomState(2)  #
rng3 = np.random.RandomState(3)  #

z = 0.5  # distance between array and source plane

# create random variable with rayleigh distribution to sample rms value
rayleigh_dist = scipy.stats.rayleigh(scale=5.0)
# create normal distribution to sample source positions
normal_dist = scipy.stats.norm(loc=0, scale=0.1688)

mg = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')

# create white noise signals and pointsources
wn_list = []
ps_list = []
for i in range(10):
    wn_list.append(ac.WNoiseGenerator(sample_freq=51200, seed=10 + i, rms=1.0, num_samples=51200))
    ps_list.append(ac.PointSource(signal=wn_list[i], mics=mg, loc=(0.0, 0.0, z)))

sm = ac.SourceMixer(sources=ps_list)
ps = ac.PowerSpectra(source=sm, block_size=512, window='Hanning')
rg = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=z, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg, steer_type='true location')
bb = ac.BeamformerBase(freq_data=ps, steer=st)


# create sampler object to sample rms value with rayleigh distribution
rms_sampling = NumericAttributeSampler(random_var=rayleigh_dist, target=wn_list, attribute='rms', random_state=rng1)

# sample PointSource positions
ps_sampling = PointSourceSampler(
    random_var=normal_dist,
    target=ps_list,
    ldir=np.array([[1.0], [1.0], [0]]),
    x_bounds=(-0.5, 0.5),
    y_bounds=(-0.5, 0.5),
    random_state=rng2,
)

# sample number of sources
src_sampling = SourceSetSampler(target=[sm], set=ps_list, random_state=rng3)

# sample five different source cases and plot beamforming result
cfreq = 2000
for _i in range(5):
    # sample
    rms_sampling.sample()
    ps_sampling.sample()
    src_sampling.sample()
    # recalculate
    pm = bb.synthetic(cfreq, 1)
    Lm = ac.L_p(pm)
    # plot
    figure()
    imshow(Lm.T, origin='lower', vmin=Lm.max() - 15, extent=rg.extend(), interpolation='bicubic')
    for src in sm.sources:
        (x, y, _) = src.loc
        plot(x, y, marker='x', color='red')
    colorbar()
    show()
