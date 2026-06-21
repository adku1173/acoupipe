"""DistributedPipeline helpers for multiprocessing tests."""

from acoular import MicGeom, PointSource, SourceMixer, WNoiseGenerator
from acoupipe.pipeline import DistributedPipeline
from acoupipe.sampler import MicGeomSampler, NumericAttributeSampler, PointSourceSampler, SourceSetSampler

from tests.constants import POS_TOTAL

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm, rayleigh


def get_distributed_pipeline(nsamples=100, num_workers=1):
    mg = MicGeom(pos_total=POS_TOTAL)
    wn_list = []
    ps_list = []
    for i in range(2):
        wn_list.append(WNoiseGenerator(sample_freq=51200, seed=100 + i, rms=1.0, num_samples=51200))
        ps_list.append(PointSource(signal=wn_list[i], mics=mg, loc=(0.0, 0.0, 1.0)))
    sm = SourceMixer(sources=ps_list)

    nas = NumericAttributeSampler(
        random_var=rayleigh(scale=5.0), target=wn_list, attribute='rms', random_state=RandomState(1)
    )
    mgs = MicGeomSampler(
        random_var=norm(loc=0, scale=0.004), ddir=np.array([[1.0], [0.5], [0]]), random_state=RandomState(2), target=mg
    )
    pss = PointSourceSampler(
        random_var=norm(loc=0, scale=0.1688),
        target=ps_list,
        ldir=np.array([[1.0], [1.0], [0.1]]),
        x_bounds=(-0.5, 0.5),
        y_bounds=(-0.5, 0.5),
        random_state=RandomState(3),
    )
    sms = SourceSetSampler(replace=False, target=[sm], set=ps_list, random_state=RandomState(4))
    pipeline = DistributedPipeline(
        sampler={1: nas, 2: mgs, 3: pss, 4: sms},
        numsamples=nsamples,
        numworkers=num_workers,
        features=lambda sampler: {'data': True},
    )
    return pipeline
