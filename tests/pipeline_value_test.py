#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:57:05 2020

@author: kujawski

provides a BasePipeline and Distributed Pipeline for testing
"""

from acoular import WNoiseGenerator, PointSource, MicGeom, SourceMixer
from scipy.stats import rayleigh, norm, poisson
from numpy.random import RandomState
from numpy.testing import assert_equal
from numpy import array
from acoupipe import BasePipeline, DistributedPipeline,\
    NumericAttributeSampler, MicGeomSampler, PointSourceSampler, SourceSetSampler
import ray


def get_pipeline(nsamples=100, mfile="./tests/array64_d0o686.xml"):
    mg = MicGeom( from_file=mfile )
    wn_list = []
    ps_list = []
    for i in range(10):
        wn_list.append(
            WNoiseGenerator(sample_freq=51200,seed=100+i, rms=1.0, numsamples=51200))
        ps_list.append(
            PointSource(signal=wn_list[i],mics=mg,loc=(0.,0.,1.0)))
    sm = SourceMixer(sources=ps_list)
    
    nas = NumericAttributeSampler(random_var=rayleigh(scale=5.), 
                                           target=wn_list, 
                                           attribute='rms',
                                           random_state=RandomState(1))

    mgs = MicGeomSampler(random_var=norm(loc=0, scale= 0.004),
                         ddir = array([[1.],[0.5],[0]]),
                         random_state= RandomState(2), 
                         target=mg)

    pss = PointSourceSampler(random_var=norm(loc=0,scale=0.1688),
                             target=ps_list,
                             ldir=array([[1.0],[1.0],[.1]]),
                             x_bounds=(-.5,.5),
                             y_bounds=(-.5,.5),
                             random_state=RandomState(3),
                             )

    sms = SourceSetSampler(replace=False,
                              target=[sm],
                              set=ps_list,
                              random_state=RandomState(4))

    pipeline = BasePipeline(sampler=[nas,mgs,pss,sms],
                            numsamples = nsamples,
                            features={
                                "num_sources" : (lambda m: len(m.sources), sm),
                                "rms_sources" : (lambda m: [s.signal.rms for s in m.sources], sm),
                                "mpos": (lambda m: m.mpos, mg),
                                "spos": (lambda m: [s.loc for s in m.sources], sm)
                                })
    return pipeline

def get_distributed_pipeline(nsamples=100,num_workers=1, mfile="./tests/array64_d0o686.xml" ):
    
    mg = MicGeom( from_file=mfile)
    wn_list = []
    ps_list = []
    for i in range(10):
        wn_list.append(
            WNoiseGenerator(sample_freq=51200,seed=100+i, rms=1.0, numsamples=51200))
        ps_list.append(
            PointSource(signal=wn_list[i],mics=mg,loc=(0.,0.,1.0)))
    sm = SourceMixer(sources=ps_list)
    
    nas = NumericAttributeSampler(random_var=rayleigh(scale=5.), 
                                           target=wn_list, 
                                           attribute='rms',
                                           random_state=RandomState(1))
    mgs = MicGeomSampler(random_var=norm(loc=0, scale= 0.004),
                         ddir = array([[1.],[0.5],[0]]),
                         random_state= RandomState(2), 
                         target=mg)
    pss = PointSourceSampler(random_var=norm(loc=0,scale=0.1688),
                             target=ps_list,
                             ldir=array([[1.0],[1.0],[.1]]),
                             x_bounds=(-.5,.5),
                             y_bounds=(-.5,.5),
                             random_state=RandomState(3),
                             )
    sms = SourceSetSampler(replace=False,
                              target=[sm],
                              set=ps_list,
                              random_state=RandomState(4))                              
    pipeline = DistributedPipeline(sampler=[nas,mgs,pss,sms],
                            numsamples = nsamples,
                            numworkers = num_workers,
                            features={
                                "num_sources" : (lambda m: len(m.sources), sm),
                                "rms_sources" : (lambda m: [s.signal.rms for s in m.sources], sm),
                                "mpos": (lambda m: m.mpos, mg),
                                "spos": (lambda m: [s.loc for s in m.sources], sm)
                                })
    return pipeline
    

        



