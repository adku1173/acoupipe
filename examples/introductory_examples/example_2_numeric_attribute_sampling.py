"""
Sampling Numeric Object Attributes
==================================

The :class:`~acoupipe.sampler.NumericAttributeSampler` class randomly varies a numeric
attribute (an ``int`` or ``float``) of one or more Acoular_ objects according to a
given probability distribution. Here it is used to randomize the root mean square
(RMS) value of a white noise signal generator.
"""

# %%
# We import Acoular_, the :class:`NumericAttributeSampler <acoupipe.sampler.NumericAttributeSampler>`, SciPy_ for the probability distribution and
# matplotlib_ for plotting.

import acoular as ac
from acoupipe.sampler import NumericAttributeSampler

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# %%
# A white noise signal generator is created. Its ``rms`` attribute is the
# value we are going to sample.

wn = ac.WNoiseGenerator(sample_freq=51200, seed=10, rms=1.0, num_samples=51200)

# %%
# The RMS values are drawn from a Rayleigh distribution. A fixed random state is
# used so that the example is reproducible.

rayleigh_dist = scipy.stats.rayleigh(scale=5.0)
rng = np.random.RandomState(1)

# %%
# The :class:`NumericAttributeSampler <acoupipe.sampler.NumericAttributeSampler>` is given the random variable, the target object(s) whose attribute
# should be manipulated, and the name of that attribute.

rms_sampling = NumericAttributeSampler(
    random_var=rayleigh_dist,
    target=[wn],
    attribute='rms',
    random_state=rng,
)

# %%
# Each call to :meth:`~acoupipe.sampler.NumericAttributeSampler.sample` draws a new
# value from the distribution and assigns it to ``wn.rms``. We repeat this many
# times and record the resulting RMS values.

rms_values = []
for _ in range(1000):
    rms_sampling.sample()
    rms_values.append(wn.rms)

# %%
# The histogram of the sampled values follows the Rayleigh probability density
# function they were drawn from.

x = np.linspace(0, max(rms_values), 200)

fig, ax = plt.subplots()
ax.hist(rms_values, bins=30, density=True, histtype='stepfilled', alpha=0.3, label='sampled RMS values')
ax.plot(x, rayleigh_dist.pdf(x), 'r-', label='Rayleigh PDF')
ax.set_xlabel('RMS value')
ax.set_ylabel('probability density')
ax.legend()
plt.show()

# %%
# Beyond this basic use, the sampler can manipulate several objects at once (pass
# a list as ``target``), assign the same drawn value to all of them
# (``equal_value=True``), sort the drawn values (``order='ascending'`` or
# ``'descending'``), normalize them so the largest equals one (``normalize=True``),
# or reject unwanted values with a ``filter`` callable.
