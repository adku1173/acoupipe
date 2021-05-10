#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:58:35 2020

@author: kujawski

Due to Acoulars caching capabilities, the pipeline can run much faster
on the second evaluation of this script

"""


from os import path
import scipy
import numpy as np
import acoular
from acoupipe import NumericAttributeSampler, BasePipeline, WriteTFRecord, \
        float_list_feature, float_feature, int64_feature

rng = np.random.RandomState(1) # scipy listens to numpy random seed (when scipy seed is None)

# create random variable with rayleigh distribution to sample rms value
rayleigh_dist = scipy.stats.rayleigh(scale=5.)

# create white noise signal
wn = acoular.WNoiseGenerator(sample_freq=51200,seed=10, rms=1.0, numsamples=51200)

# create sampler object to sample rms value with rayleigh distribution
rms_sampling = NumericAttributeSampler(random_var=rayleigh_dist, 
                                       target=[wn], 
                                       attribute='rms',
                                       random_state=rng)

# build processing chain
micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
mg = acoular.MicGeom( from_file=micgeofile )
p1 = acoular.PointSource( signal=wn,  mics=mg )
ps = acoular.PowerSpectra( time_data=p1, block_size=128, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
increment=0.01 )
st = acoular.SteeringVector( grid = rg, mics=mg )
bb = acoular.BeamformerBase( freq_data=ps, steer=st )

# create Pipeline
pipeline = BasePipeline(sampler=[rms_sampling],
                        numsamples = 5,
                        features={
                            "sourceMap" : (lambda b:b.synthetic(1000,1), bb),
                            "rms" : (lambda g: g.rms, wn) 
                            })

# create TFRecordWriter to save pipeline output to TFRecord File
writer = WriteTFRecord(source=pipeline, 
                        name="test.tfrecord",
                        encoder_funcs = {"sourceMap": float_list_feature,
                                        "rms": float_feature,
                                        "idx": int64_feature},
                        verbosity=1)
writer.save() # save data to file
