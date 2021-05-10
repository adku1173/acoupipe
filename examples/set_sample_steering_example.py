#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:51:11 2021

@author: Jekosch
"""

import numpy as np
import acoular
acoular.config.global_caching = "none"
from acoular import WNoiseGenerator, PointSource, SourceMixer, MicGeom
from acoupipe import SetSampler
from pylab import figure, plot, imshow, colorbar, show, title

# define random state
rng =  np.random.RandomState(seed=1) 

z = .5 # distance between array and source plane
mg = MicGeom( from_file="array64_d0o686.xml" )

# create white noise signals and pointsources
wn1 = WNoiseGenerator(sample_freq=51200,seed=10, rms=1.0, numsamples=51200)
ps1 = PointSource(signal=wn1,mics=mg,loc=(0.2,0.2,z))

wn2 = WNoiseGenerator(sample_freq=51200,seed=10, rms=1.0, numsamples=51200)
ps2 = PointSource(signal=wn2,mics=mg,loc=(-0.2,-0.3,z))

sm = SourceMixer(sources=[ps1,ps2])
ps = acoular.PowerSpectra( time_data=sm, block_size=512, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=z, \
increment=0.01 )
st = acoular.SteeringVector( grid = rg, mics=mg, steer_type='true location' )
bb = acoular.BeamformerBase( freq_data=ps, steer=st)


#sample from 4 different sterring vectors
steer_set = ['true level', 'true location', 'classic', 'inverse',] 

steer_sampling = SetSampler(
    target = [st],
    attribute="steer_type", 
    set=steer_set,
    single_value = False,
    replace=False,
    prob_list =  [0.3 , 0.3, 0.2, 0.2],
    random_state= rng
    )


cfreq = 2000
for i in range(10):
    # sample
    steer_sampling.sample()
    # recalculate
    pm = bb.synthetic( cfreq, 1 )
    Lm = acoular.L_p( pm )
    # plot
    figure()
    imshow( Lm.T, origin = "lower", vmin=Lm.max()-15, extent=rg.extend(), \
    interpolation='bicubic')
    title(st.steer_type)
    for src in sm.sources:
        (x,y,_) = src.loc
        plot(x,y,marker="x",color="red")
    colorbar()
    show()





