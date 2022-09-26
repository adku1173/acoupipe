
import ray
import acoular
import numpy as np
from os import path
from scipy.stats import poisson, norm 
from acoular import config, MicGeom, WNoiseGenerator, PointSource, SourceMixer,\
    PowerSpectra, MaskedTimeInOut, Environment
from acoupipe import MicGeomSampler, PointSourceSampler, SourceSetSampler, \
    NumericAttributeSampler, ContainerSampler, DistributedPipeline, BasePipeline,get_frequency_index_range
from acoupipe import RefSourceMapFeature, get_source_loc, get_source_p2, CSMFeature, SourceMapFeature,\
    NonRedundantCSMFeature, SpectrogramFeature, RefSBL
from acoupipe import set_pipeline_seeds
from spectacoular import SpectraInOut
from acoupipe import Options # sbl module
R = 0.05

def simulate(
    dataset,
    numsamples,
    features,
    nsources=None,
    freq=None,
    freq_index=None,
    num=0,
    tasks=1,
    startsample=1,
    cache_dir = "./datasets",
    cache_bf = False,
    cache_csm = False
    ):

    if freq and freq_index:
        raise ValueError("It is only allowed to set either 'freq' or 'freq_index', not both at the same time!")

    dirpath = path.dirname(path.abspath(__file__))
    # Fixed Parameters
    C = 343. # speed of sound
    HE = 40 # Helmholtz number (defines the sampling frequency) 
    SFREQ = HE*C # /ap with ap=1.0
    BLOCKSIZE = 128 # block size used for FFT 
    SIGLENGTH=5 # length of the simulated signal
    MFILE = path.join(dirpath,"tub_vogel64_ap1.xml") # Microphone Geometry
    REF_MIC = 63 # index of the reference microphone 

    if nsources:
        MAXSRCS = nsources # maximum number of sources
    else:    
        MAXSRCS = 10 # maximum number of sources
    # Random Variables
    mic_rvar = norm(loc=0, scale=0.001) # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
    pos_rvar = norm(loc=0,scale=0.1688) # source positions
    nsrc_rvar = poisson(mu=3,loc=1) # number of sources

    # Acoular Config
    config.h5library = 'h5py'
    if cache_bf or cache_csm:
        config.cache_dir = path.join(cache_dir,'cache') # set up cache file dir
    print("cache file directory at: ",config.cache_dir)


    # Computational Pipeline Acoular
    # Microphone Geometry
    mg_manipulated = MicGeom(from_file=MFILE) 
    mg_fixed = MicGeom(from_file=MFILE)
    # Environment
    env = Environment(c=C)
    # Signals
    white_noise_signals = [
        WNoiseGenerator(sample_freq=SFREQ,seed=i+1,numsamples=SIGLENGTH*SFREQ) for i in range(MAXSRCS)
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
    ps_args = {'block_size':BLOCKSIZE, 'overlap':'50%', 'window':"Hanning", 'precision':'complex64'}
    ps_csm = PowerSpectra(time_data=sources_mix,cached=cache_csm,**ps_args)
    ps_ref = PowerSpectra(**ps_args,cached=False) # caching takes more time than calculation for a single channel
    spectra_inout = SpectraInOut(source=sources_mix,block_size=BLOCKSIZE, window="Hanning",overlap="50%" )

    ps_ref.time_data = MaskedTimeInOut(source=sources_mix,invalid_channels=[_ for _ in range(64) if not _  == REF_MIC]) # masking other channels than the reference channel

    # Set up Beamformer object to calculate sourcemap feature
    if ("sourcemap" in features) or ("ref_cleansc" in features) or ("SBL" in features):
        bb_args = {'r_diag':False,}
        sv_args = {'steer_type':'true level', 'ref':mg_fixed.mpos[:,REF_MIC]}
        rg = acoular.RectGrid(
                        x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=.5,increment=1/63) # 64 x 64 grid           
        st = acoular.SteeringVector(
                        grid=rg, mics=mg_fixed, env=env, **sv_args)
        bb = acoular.BeamformerBase(
                        freq_data=ps_csm, steer=st, cached=cache_bf, precision='float32',**bb_args)
        bfcleansc = acoular.BeamformerCleansc(
                        freq_data=ps_csm, steer=st, cached=cache_bf, precision='float32',**bb_args)

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
    if nsources: src_sampling.numsamples = nsources

    rms_sampling = ContainerSampler(
                        random_func=sample_rms)

    if not nsources: # if no number of sources is specified, the number of sources will be samples randomly
        nsrc_sampling =  NumericAttributeSampler(
                            random_var=nsrc_rvar, 
                            target=[src_sampling], 
                            attribute='numsamples',
                            filter=lambda x: x<=MAXSRCS,
                            )

        sampler_list = [mic_sampling, nsrc_sampling, src_sampling, rms_sampling, pos_sampling]
    else:
        sampler_list = [mic_sampling, src_sampling, rms_sampling, pos_sampling]
    
    if tasks > 1:
        ray.init()
        pipeline = DistributedPipeline(
                        sampler=sampler_list,
                        numworkers=tasks,
                        )    
    else:
        pipeline = BasePipeline(
                        sampler=sampler_list,
                        )

    # desired frequency/helmholtz number
    if freq != None:
        fidx = [get_frequency_index_range(ps_csm.fftfreq(),f,num) for f in freq]
        ps_csm.ind_low = min([f[0] for f in fidx])
        ps_csm.ind_high = max([f[1] for f in fidx])
    elif freq_index:
        fftfreq=ps_csm.fftfreq()
        freq = np.array([fftfreq[i] for i in freq_index])
        fidx = [get_frequency_index_range(fftfreq,f,num) for f in freq]
        ps_csm.ind_low = min([f[0] for f in fidx])
        ps_csm.ind_high = max([f[1] for f in fidx])
    else:
        fidx = None
        freq = None
        ps_csm.ind_low = 1
        ps_csm.ind_high = -1
        print(freq,freq_index)
    print(fidx)
    # set up the feature dict with methods to get the labels
    feature_methods = {
        "loc": (get_source_loc, sources_mix, MAXSRCS), # (callable, arg1, arg2, ...)
        "nsources": (lambda smix: len(smix.sources), sources_mix),
        "p2": (get_source_p2, sources_mix, ps_ref, fidx, MAXSRCS, config.cache_dir),
    }

    feature_objects = []
    if "csm" in features:
        feature = CSMFeature(feature_name="csm",
                        power_spectra=ps_csm,
                        fidx=fidx,
                        cache_dir=config.cache_dir   
                        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)

    if "csmtriu" in features:
        feature = NonRedundantCSMFeature(feature_name="csmtriu",
                        power_spectra=ps_csm,
                        fidx=fidx,
                        cache_dir=config.cache_dir   
                        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)

    if "sourcemap" in features:
        feature = SourceMapFeature(feature_name="sourcemap",
                        beamformer=bb,
                        f=freq,
                        num=num,
                        cache_dir=config.cache_dir   
                        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)    

    if "ref_cleansc" in features:
        feature = RefSourceMapFeature(feature_name="ref_cleansc",
                        beamformer=bfcleansc,
                        sourcemixer=sources_mix,
                        powerspectra=ps_ref,
                        r=0.05,
                        f=freq,
                        num=num,
                        cache_dir=config.cache_dir   
                        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)    

    if "spectrogram" in features:
        feature = SpectrogramFeature(feature_name="spectrogram",
                        spectra_inout=spectra_inout,
                        fidx=fidx,
                        cache_dir=config.cache_dir   
                        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)

    if "SBL" in features:
        feature = RefSBL(
            feature_name = "SBL",
            spectra_inout = spectra_inout,
            steer = st,
            #options = Options(convergence_error=10 ** (-8), gamma_range=10 ** (-4), convergence_maxiter=5000, convergence_min_iteration=1, status_report=1, fixedpoint=1, Nsource=1, flag=0),
            fidx = fidx, 
        )
        feature_methods = feature.add_feature_funcs(feature_methods)
        feature_objects.append(feature)

    # add features to the pipeline
    #print(feature_methods)
    pipeline.features = feature_methods
    set_pipeline_seeds(pipeline, startsample, numsamples, dataset)

    # yield the data    
    for data in pipeline.get_data():
        yield data
        
if __name__ == "__main__":
    validation_generator = simulate(dataset="validation",numsamples=1,features="SBL",num=3,freq=[1000,2000])
    data = next(validation_generator)