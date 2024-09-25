"""
Demonstrates the usage of the acoupipe module to build a processing pipeline.

Due to Acoulars caching capabilities, the pipeline can run much faster
on the second evaluation of this script. 
"""
import sys
from os import path

import acoular
import numpy as np
import scipy

from acoupipe.loader import LoadH5Dataset
from acoupipe.pipeline import BasePipeline
from acoupipe.sampler import NumericAttributeSampler
from acoupipe.writer import WriteH5Dataset

   # float_list_feature, float_feature, int64_feature #WriteTFRecord

####################global######################
acoular.config.h5library = "h5py"
acoular.config.global_caching = "none"
DATASET_NAME = "example_hdfdata_format.h5"
################################################

rng = np.random.RandomState(1) # scipy listens to numpy random seed (when scipy seed is None)

# create random variable with rayleigh distribution to sample rms value
rayleigh_dist = scipy.stats.rayleigh(scale=5.)

# create white noise signal
wn = acoular.WNoiseGenerator(sample_freq=51200,seed=10, rms=1.0, numsamples=51200)

# create sampler object to sample rms value with rayleigh distribution
rms_sampling = NumericAttributeSampler(random_var=rayleigh_dist,
                                       target=[wn],
                                       attribute="rms",
                                       random_state=rng)

# build processing chain
micgeofile = path.join(path.split(acoular.__file__)[0],"xml","array_64.xml")
mg = acoular.MicGeom( from_file=micgeofile )
p1 = acoular.PointSource( signal=wn,  mics=mg )
ps = acoular.PowerSpectra( time_data=p1, block_size=128, window="Hanning" )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
increment=0.01 )
st = acoular.SteeringVector( grid = rg, mics=mg )
bb = acoular.BeamformerBase( freq_data=ps, steer=st )


# create a feature extraction function
def extract_features(sampler, beamformer, whitenoise):
    """Extract features from sampler object."""
    return {"rms" : wn.rms, "sourcemap" : beamformer.synthetic(1000,1)}

# create Pipeline
pipeline = BasePipeline(sampler={1: rms_sampling},
                        numsamples = 1,
                        features=(extract_features, bb, wn))

#%% save data to file

# One can write the data to multiple files by chaining multiple Writer Objects

# create some additional meta data
metadata = {"sample_freq": 51200,
            "freq" : 1000,
            "bandwidth" : "octave",
            "block_size" : 128 }

#create TFRecordWriter to save pipeline output to TFRecord File
writer_sourcemap = WriteH5Dataset(source=pipeline,
                        name="sourcemap_"+DATASET_NAME,
                        features=["sourcemap", "rms"],
                        metadata=metadata)

writer_rms = WriteH5Dataset(source=writer_sourcemap,
                        name="rms_"+DATASET_NAME,
                        features=["rms"])

# #use the given save method
# writer_rms.save() # save data to file

#alternatively use the result method
for d in writer_rms.get_data():
    print(d)

# import tables as pt
# pfile = pt.File("_"+DATASET_NAME,"w")


#%% load data from file

loader_sourcemap = LoadH5Dataset(name="sourcemap_"+DATASET_NAME)
loader_rms = LoadH5Dataset(name="rms_"+DATASET_NAME)
print(loader_sourcemap.numsamples)
print(loader_sourcemap.numfeatures)


#%% Alternatively one can write the data to a TFRecord file

try:
    from acoupipe.writer import WriteTFRecord, float_feature, float_list_feature
except ImportError:
    sys.exit()

DATASET_NAME = "test_hdfdata_format.tfrecord"

#create TFRecordWriter to save pipeline output to TFRecord File
writer_sourcemap = WriteTFRecord(source=pipeline,
                        name="sourcemap_"+DATASET_NAME,
                        encoder_funcs={"sourcemap":float_list_feature})

writer_rms = WriteTFRecord(source=writer_sourcemap,
                        name="rms_"+DATASET_NAME,
                        encoder_funcs={"rms_":float_feature},)

writer_rms.save()

for d in writer_rms.get_data():
    print(d)
