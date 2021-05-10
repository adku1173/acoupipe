#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:53:22 2020

@author: kujawski
"""

import scipy.stats 
from acoular import WNoiseGenerator
from acoupipe import NumericAttributeSampler

# create white noise signal
wn = WNoiseGenerator(sample_freq=51200,seed=10, rms=1.0, numsamples=51200)

# create random variable with rayleigh distribution to sample rms value
rayleigh_dist = scipy.stats.rayleigh(scale=5.)

# create sampler object to sample rms value with rayleigh distribution
rms_sampling = NumericAttributeSampler(random_var=rayleigh_dist, 
                                       target=[wn], 
                                       attribute='rms')

rms_values = []
for i in range(1000):
    rms_sampling.sample()
    rms_values.append(wn.rms)
    

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.hist(rms_values, density=True, histtype='stepfilled', alpha=0.2)
plt.show()
