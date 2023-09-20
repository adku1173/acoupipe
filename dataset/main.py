import ray
import argparse
import logging
import numpy as np
from os import path
from scipy.stats import poisson, norm 
from acoular import config, MicGeom, WNoiseGenerator, PointSource, SourceMixer,\
    PowerSpectra, MaskedTimeInOut, Environment, RectGrid, SteeringVector, BeamformerBase, BeamformerCleansc
from acoupipe import MicGeomSampler, PointSourceSampler, SourceSetSampler, \
    NumericAttributeSampler, ContainerSampler, DistributedPipeline, BasePipeline,\
        WriteTFRecord, WriteH5Dataset, float_list_feature, int_list_feature,\
            int64_feature
from features import RefSourceMapFeature, get_source_loc, get_source_p2, CSMFeature, SourceMapFeature,\
    NonRedundantCSMFeature
from helper import set_pipeline_seeds, set_filename
from constants import ds1_constants
import numba

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs="+", default=["training","validation"], choices=["training", "validation"],
                    help="Whether to compute both data sets ('training validation') or only the 'training' / 'validation' data set. Defaults to compute training and validation data set")
parser.add_argument('--tsamples', type=int, default=500000,
                    help="Total number of  training samples to simulate. Default : 500,000")
parser.add_argument('--tstart', type=int, default=1,
                    help="Start simulation at a specific sample of the data set. Default: 1")                    
parser.add_argument('--vsamples', type=int, default=10000,
                    help="Total number of validation samples to simulate. Default : 10,000")
parser.add_argument('--vstart', type=int, default=1,
                    help="Start simulation at a specific sample of the data set. Default: 1")                          
parser.add_argument('--tpath', type=str, default="./datasets",
                    help="Path of simulated training data. Default is the current working directory")
parser.add_argument('--vpath', type=str, default="./datasets",
                    help="Path of simulated validation data. Default is the current working directory")
parser.add_argument('--file_format', type=str, default="h5", choices=["tfrecord", "h5"],
                    help="Desired file format to store the data sets.")
parser.add_argument('--cache_dir', type=str, default=None,
                    help="Path of cached data. Needs to be specified in case when --cache_bf or --cache_csm flag is used.")
parser.add_argument('--freq_index', type=int, default=None,
                    help="Returns only the features and targets for the specified frequency index. Default is 'None' (all frequencies will be calculated and included in the data set)")
parser.add_argument('--nsources', type=int, default=None,
                    help="Calculates the data set with a fixed number of sources. Default is 'None', meaning that the number of sources present will be sampled randomly.")
parser.add_argument('--features', nargs="+", default=["csm"], choices=["sourcemap", "csmtriu", "csm", "ref_cleansc"],
                    help="Whether to compute data set containing the csm or the beamforming map as the main feature. Default is 'csm'")
parser.add_argument('--tasks', type=int, default=1,
                    help="Number of asynchronous tasks. Defaults to '1' (non-distributed)")
parser.add_argument('--head', type=str, default=None,
                    help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.") 
parser.add_argument('--cache_csm', action="store_true",
                    help="Caches the results of the CSM calculation when added as an argument") 
parser.add_argument('--cache_bf', action="store_true",
                    help="Caches the results of the beamformer calculation when added as an argument. Only relevant if 'sourcemap' is included in --features list.")                     
parser.add_argument('--log', action="store_true",
                    help="Whether to log timing statistics to file. Only for internal use.")                          
args = parser.parse_args()
#args = parser.parse_args(["--datasets=training","--tsamples=1","--file_format=h5","--freq_index=10"])

# use h5py package
config.h5library = 'h5py'
if args.cache_bf or args.cache_csm:
    if not args.cache_dir:
        raise ValueError("Please specify a cache_dir via the --cache_dir flag!")
    config.cache_dir = path.join(args.cache_dir,'cache') # set up cache file dir
    print("cache file directory at: ",config.cache_dir)
    cache_dir = config.cache_dir
else: # nothing is cached
    cache_dir = None

# Acoular Config
num_threads = numba.get_num_threads()

# Ray Config
if args.tasks > 1:
    ray.init(address=args.head)
    num_threads=1

# logging for debugging and timing statistic purpose
if args.log:
    logging.basicConfig(level=logging.INFO) # root logger
    logger = logging.getLogger()

dirpath = path.dirname(path.abspath(__file__))
# Fixed Parameters

if args.nsources:
    MAXSRCS = args.nsources # maximum number of sources
else:    
    MAXSRCS = 10 # maximum number of sources

# Random Variables
mic_rvar = norm(loc=0, scale=0.001) # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
pos_rvar = norm(loc=0,scale=0.1688) # source positions
nsrc_rvar = poisson(mu=3,loc=1) # number of sources

# Computational Pipeline Acoular
# Microphone Geometry
mg_manipulated = MicGeom(from_file=ds1_constants['MFILE']) 
mg_fixed = MicGeom(from_file=ds1_constants['MFILE'])
# Environment
env = Environment(c=ds1_constants['C'])
# Signals
white_noise_signals = [
    WNoiseGenerator(sample_freq=ds1_constants['SFREQ'],seed=i+1,numsamples=ds1_constants['SIGLENGTH']*ds1_constants['SFREQ']) for i in range(MAXSRCS)
    ] 
# Monopole sources emitting the white noise signals
point_sources = [
    PointSource(signal=signal,mics=mg_manipulated,env=env, loc=(0,0,.5)) for signal in white_noise_signals
    ]
# Source Mixer mixing the signals of all sources (number will be sampled)
sources_mix = SourceMixer(sources=point_sources)

# Set up PowerSpectra objects to calculate CSM feature and reference p2 value
# first object is used to calculate the full CSM 
# second object will be used to calculate the p2 value at the reference microphone (for each present source)
ps_args = {'block_size':ds1_constants['BLOCKSIZE'], 'overlap':ds1_constants['OVERLAP'], 'window':ds1_constants['WINDOW'], 'precision':'complex64'}
ps_csm = PowerSpectra(time_data=sources_mix,cached=args.cache_csm,**ps_args)
ps_ref = PowerSpectra(**ps_args,cached=False) # caching takes more time than calculation for a single channel
ps_ref.time_data = MaskedTimeInOut(source=sources_mix,invalid_channels=[_ for _ in range(64) if not _  == ds1_constants['REF_MIC']]) # masking other channels than the reference channel

# Set up Beamformer object to calculate sourcemap feature
if ("sourcemap" in args.features) or ("ref_cleansc" in args.features):
    bb_args = {'r_diag':True,}
    sv_args = {'steer_type':'true level', 'ref':mg_fixed.mpos[:,ds1_constants['REF_MIC']]}
    rg = RectGrid(
                    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=.5,increment=1/63) # 64 x 64 grid           
    st = SteeringVector(
                    grid=rg, mics=mg_fixed, env=env, **sv_args)
    bb = BeamformerBase(
                    freq_data=ps_csm, steer=st, cached=args.cache_bf, precision='float32',**bb_args)
    bfcleansc = BeamformerCleansc(
                    freq_data=ps_csm, steer=st, cached=args.cache_bf, precision='float32',**bb_args)

# Computational Pipeline AcouPipe 

# callable function to draw and assign sound pressure RMS values to the sources of the SourceMixer object
def sample_rms(rng):
    "draw source pressures square, Rayleigh distribution, sort them, calc rms"
    nsrc = len(sources_mix.sources)
    p_rms = np.sqrt(np.sort(rng.rayleigh(5,nsrc))[::-1]) # draw source pressures square, Rayleigh distribution, sort them, calc rms
    p_rms /= p_rms.max() #norm it
    for i, rms in enumerate(p_rms):
        sources_mix.sources[i].signal.rms = rms # set rms value

mic_sampling = MicGeomSampler(
                    random_var=mic_rvar,
                    target=mg_manipulated,
                    ddir=np.array([[1.0],[1.0],[0]])
                    ) # ddir along two dimensions -> bivariate sampling

pos_sampling = PointSourceSampler(
                    random_var=pos_rvar,
                    target=sources_mix.sources,
                    ldir=np.array([[1.0],[1.0],[0.0]]), # ldir: 1.0 along first two dimensions -> bivariate sampling
                    x_bounds=(-.5,.5), # only allow values between -.5 and .5
                    y_bounds=(-.5,.5),
                    )

src_sampling = SourceSetSampler(    
                    target=[sources_mix],
                    set=point_sources,
                    replace=False,
                    numsamples=3,
                    ) # draw point sources from point_sources set (number of sources is sampled by nrcs_sampling object)
if args.nsources: src_sampling.numsamples = args.nsources

rms_sampling = ContainerSampler(
                    random_func=sample_rms)

if not args.nsources: # if no number of sources is specified, the number of sources will be samples randomly
    nsrc_sampling =  NumericAttributeSampler(
                        random_var=nsrc_rvar, 
                        target=[src_sampling], 
                        attribute='numsamples',
                        filter=lambda x: x<=MAXSRCS,
                        )

    sampler_list = [mic_sampling, nsrc_sampling, src_sampling, rms_sampling, pos_sampling]
else:
    sampler_list = [mic_sampling, src_sampling, rms_sampling, pos_sampling]
 
if args.tasks > 1:
    pipeline = DistributedPipeline(
                    sampler=sampler_list,
                    numworkers=args.tasks,
                    )    
else:
    pipeline = BasePipeline(
                    sampler=sampler_list,
                    )

# desired frequency/helmholtz number
fidx = args.freq_index
if fidx:
    freq = ps_csm.fftfreq()[fidx]
    he = freq/ds1_constants['C']
    ps_csm.ind_low = fidx
    ps_csm.ind_high = fidx+1
    freq_str=f"he{he}-{freq}Hz"
else:
    freq = None
    he = None
    ps_csm.ind_low = 1
    ps_csm.ind_high = -1
    freq_str=f"fullfreq"

# set up the feature dict with methods to get the labels
feature_methods = {
    "loc": (get_source_loc, sources_mix, MAXSRCS), # (callable, arg1, arg2, ...)
    "nsources": (lambda smix: len(smix.sources), sources_mix),
    "p2": (get_source_p2, sources_mix, ps_ref, fidx, MAXSRCS, cache_dir),
}

feature_objects = []
if "csm" in args.features:
    feature = CSMFeature(feature_name="csm",
                     power_spectra=ps_csm,
                     fidx=fidx,
                     cache_dir=cache_dir   
                    )
    feature_methods = feature.add_feature_funcs(feature_methods)
    feature_objects.append(feature)

if "csmtriu" in args.features:
    feature = NonRedundantCSMFeature(feature_name="csmtriu",
                     power_spectra=ps_csm,
                     fidx=fidx,
                     cache_dir=cache_dir   
                    )
    feature_methods = feature.add_feature_funcs(feature_methods)
    feature_objects.append(feature)

if "sourcemap" in args.features:
    feature = SourceMapFeature(feature_name="sourcemap",
                     beamformer=bb,
                     f=freq,
                     num=0,
                     cache_dir=cache_dir   
                    )
    feature_methods = feature.add_feature_funcs(feature_methods)
    feature_objects.append(feature)    

if "ref_cleansc" in args.features:
    feature = RefSourceMapFeature(feature_name="ref_cleansc",
                     beamformer=bfcleansc,
                     sourcemixer=sources_mix,
                     powerspectra=ps_ref,
                     r=0.05,
                     f=freq,
                     num=0,
                     cache_dir=cache_dir   
                    )
    feature_methods = feature.add_feature_funcs(feature_methods)
    feature_objects.append(feature)    

# add features to the pipeline
pipeline.features = feature_methods

# for metadata
if not args.freq_index:
    freq = ps_csm.fftfreq()

metadata = {
    'VERSION' : ds1_constants['VERSION'],
    'Helmholtz number' : freq/ds1_constants['C'],
    'frequency': freq,
    'mic_geometry' : mg_fixed.mpos_tot.copy(),
    'reference_mic_index' : ds1_constants['REF_MIC'],
    'c' : ds1_constants['C'],
    'sample_freq': ds1_constants['SFREQ'],
    **ps_args # add power spectra arguments to metadata
}

# compute the data sets
for dataset in args.datasets:
    if dataset == "training":
        start_sample = args.tstart
        samples = args.tsamples
        dpath = args.tpath
    elif dataset == "validation":
        start_sample = args.vstart
        samples = args.vsamples
        dpath = args.vpath
    set_pipeline_seeds(pipeline, start_sample, samples, dataset)
    
    # Create chain of writer objects to write data sets to file
    # Individual files will be written for all features 
    source=pipeline
    for feature in feature_objects:
        if args.file_format == "tfrecord":
            # set up encoder functions to write to .tfrecord
            encoder_dict = {
                            "loc": float_list_feature,
                            "p2": float_list_feature,
                            "nsources": int64_feature,
                            "idx": int64_feature,
                            "seeds": int_list_feature,
                            }
            # create TFRecordWriter to save pipeline output to TFRecord File
            writer = WriteTFRecord(source=source,
                                    encoder_funcs=feature.add_encoder_funcs(encoder_dict))
        elif args.file_format == "h5":
            feature_names=["loc","p2","nsources","idx","seeds"]
            writer = WriteH5Dataset(source=source,
                                    features=feature.add_feature_names(feature_names),
                                    metadata=feature.add_metadata(metadata.copy()),
                                    )    
        set_filename(writer,dpath,*[dataset,f"{start_sample}-{start_sample+samples-1}"]+[feature.feature_name]+[f"{MAXSRCS}src",freq_str,ds1_constants['VERSION']])
        source=writer
    # for debugging and timing statistics
    if args.log:
        pipeline_log = logging.FileHandler(".".join(writer.name.split('.')[:-1]) + ".log",mode="w") # log everything to file
        pipeline_log.setFormatter(logging.Formatter('%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d-%(message)s', datefmt='%Y-%m-%d,%H:%M:%S'))
        logger.addHandler(pipeline_log) # attach handler to the root logger

    # start calculation
    writer.save() # start the calculation

    # remove handler
    if args.log:
        logger.removeHandler(pipeline_log)

