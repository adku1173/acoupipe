"""BasePipeline and Distributed Pipeline for testing."""

from acoular import MicGeom, PointSource, SourceMixer, WNoiseGenerator
from numpy import array
from numpy.random import RandomState
from scipy.stats import norm, rayleigh

from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.sampler import MicGeomSampler, NumericAttributeSampler, PointSourceSampler, SourceSetSampler

mpos_tot = array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])

def get_pipeline(nsamples):
    mg = MicGeom( mpos_tot = mpos_tot )
    wn_list = []
    ps_list = []
    for i in range(2):
        wn_list.append(
            WNoiseGenerator(sample_freq=51200,seed=100+i, rms=1.0, numsamples=51200))
        ps_list.append(
            PointSource(signal=wn_list[i],mics=mg,loc=(0.,0.,1.0)))
    sm = SourceMixer(sources=ps_list)
    
    nas = NumericAttributeSampler(random_var=rayleigh(scale=5.), 
                                           target=wn_list, 
                                           attribute="rms",
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

    pipeline = BasePipeline(sampler={1:nas,2:mgs,3:pss,4:sms},
                            numsamples = nsamples,
                            features = lambda sampler: {"data":True})
    return pipeline

def get_distributed_pipeline(nsamples=100,num_workers=1):
    
    mg = MicGeom( mpos_tot = mpos_tot )
    wn_list = []
    ps_list = []
    for i in range(2):
        wn_list.append(
            WNoiseGenerator(sample_freq=51200,seed=100+i, rms=1.0, numsamples=51200))
        ps_list.append(
            PointSource(signal=wn_list[i],mics=mg,loc=(0.,0.,1.0)))
    sm = SourceMixer(sources=ps_list)
    
    nas = NumericAttributeSampler(random_var=rayleigh(scale=5.), 
                                           target=wn_list, 
                                           attribute="rms",
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
    pipeline = DistributedPipeline(sampler={1:nas,2:mgs,3:pss,4:sms},
                            numsamples = nsamples,
                            numworkers = num_workers,
                            features = lambda sampler: {"data":True})
    return pipeline
    

        



