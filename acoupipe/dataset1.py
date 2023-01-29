from copy import deepcopy
from datetime import datetime
from os import path
from functools import partial
import numpy as np
import ray
from acoular import (
    BeamformerBase,
    BeamformerCleansc,
    Environment,
    MaskedTimeInOut,
    MicGeom,
    PointSource,
    PowerSpectra,
    RectGrid,
    SourceMixer,
    SteeringVector,
    WNoiseGenerator,
    SamplesGenerator,
)
from scipy.stats import norm, poisson

from .config import TF_FLAG
from .features import CSMFeature, EigmodeFeature, NonRedundantCSMFeature, SourceMapFeature, get_source_p2
from .helper import _handle_cache, _handle_log, get_frequency_index_range, set_pipeline_seeds
from .pipeline import BasePipeline, DistributedPipeline
from .sampler import ContainerSampler, MicGeomSampler, NumericAttributeSampler, PointSourceSampler, SourceSetSampler
from .writer import WriteH5Dataset

if TF_FLAG:
    from .writer import WriteTFRecord, float_list_feature, int64_feature, int_list_feature

from acoupipe import __file__ as acoupipe_path

VERSION = "ds1-v01"
DEFAULT_ENV = Environment(c=343.)
DEFAULT_MICS = MicGeom(from_file=path.join(path.split(acoupipe_path)[0], "xml", "tub_vogel64_ap1.xml"))
DEFAULT_GRID = RectGrid(y_min=-.5,y_max=.5,x_min=-.5,x_max=.5,z=.5,increment=1/63)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")
DEFAULT_STEER = SteeringVector(grid=DEFAULT_GRID, mics=DEFAULT_MICS, env=DEFAULT_ENV, steer_type ="true level")
DEFAULT_FREQ_DATA = PowerSpectra(time_data=SamplesGenerator(sample_freq=40*343), block_size=128, overlap="50%", window="Hanning", precision="complex64")
DEFAULT_RANDOM_VAR = {
            "mic_rvar" : norm(loc=0, scale=0.001), # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
            "loc_rvar" : norm(loc=0, scale=0.1688*DEFAULT_MICS.aperture),  # source positions
            "nsrc_rvar" : poisson(mu=3, loc=1)  # number of sources
        }

class Dataset1:

    def __init__(
            self, 
            size, 
            features, 
            f=None, 
            num=0, 
            fs=40*343.,
            max_nsources = 10,
            min_nsources = 1,
            steer = DEFAULT_STEER,
            beamformer = DEFAULT_BEAMFORMER,
            freq_data = DEFAULT_FREQ_DATA,
            random_var = DEFAULT_RANDOM_VAR):       
        self.size = size
        self.features = features
        self.f = f
        self.num = num
        self.fs = fs
        self.max_nsources = max_nsources
        self.min_nsources = min_nsources
        self.steer = steer
        self.beamformer = beamformer
        self.freq_data = freq_data
        self.random_var = random_var

    def _get_freq_indices(self):
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                self.f = [self.f]
            fidx = [get_frequency_index_range(
                self.freq_data.fftfreq(), f_, self.num) for f_ in self.f]
        else:
            fidx = None
        return fidx

    def build_pipeline(self, parallel, cache_csm, cache_bf, cache_dir):
        # sets the reference microphone
        ref_mic = np.argmin(np.linalg.norm((self.steer.mics.mpos - self.steer.mics.center[:,np.newaxis]),axis=0))
        self.steer.ref = self.steer.mics.mpos[:,ref_mic]
        # create copy for noisy positions
        self.noisy_mics = deepcopy(self.steer.mics) # Microphone geometry with positional noise
        white_noise_signals = [ # create source signal array
            WNoiseGenerator(sample_freq=self.fs, seed=i+1, numsamples=5*self.fs) for i in range(self.max_nsources)
        ]  # Signals
        self.point_sources = [
            PointSource(
                signal=signal, mics=self.noisy_mics, env=self.steer.env, loc=(0, 0, self.steer.grid.z)) for signal in white_noise_signals
        ]  # Monopole sources emitting the white noise signals
        self.sources_mix = SourceMixer(sources=self.point_sources) # Source Mixer mixing the signals of all sources
        self.freq_data.time_data = self.sources_mix
        self.source_freq_data = deepcopy(self.freq_data) # will be used to calculate the p2 value at the reference microphone 
        self.source_freq_data.cached = False
        self.source_freq_data.time_data = MaskedTimeInOut(source=self.sources_mix, invalid_channels=[_ for _ in range(
            self.steer.mics.num_mics) if not _ == ref_mic]) # mask all channels except ref mic       
        # set up the feature dict with methods to get the labels
        fidx = self._get_freq_indices()
        features = {  # (callable, arg1, arg2, ...)
            "loc": (lambda smix: np.array([s.loc for s in smix.sources], dtype=np.float32).T, self.sources_mix),
            "p2": (get_source_p2, self.sources_mix, self.source_freq_data, fidx, None),
        }
        features.update(
            self.setup_input_features(cache_csm, cache_bf, cache_dir))
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=self.setup_sampler(),features=features)

    def setup_sampler(self):
        # callable function to draw and assign sound pressure RMS values to the sources of the SourceMixer object
        def sample_rms(rng):
            """Draw source pressures square, Rayleigh distribution, sort them, calc rms."""
            nsrc = len(self.sources_mix.sources)
            # draw source pressures square, Rayleigh distribution, sort them, calc rms
            p_rms = np.sqrt(np.sort(rng.rayleigh(5, nsrc))[::-1])
            p_rms /= p_rms.max()  # norm it
            for i, rms in enumerate(p_rms):
                self.sources_mix.sources[i].signal.rms = rms  # set rms value

        mic_sampling = MicGeomSampler(
            random_var=self.random_var["mic_rvar"],
            target=self.noisy_mics,
            ddir=np.array([[1.0], [1.0], [0]])
        )  # ddir along two dimensions -> bivariate sampling

        pos_sampling = PointSourceSampler(
            random_var=self.random_var["loc_rvar"],
            target=self.sources_mix.sources,
            # ldir: 1.0 along first two dimensions -> bivariate sampling
            ldir=np.array([[1.0], [1.0], [0.0]]),
            # only allow values in the observation area
            x_bounds=(self.steer.grid.x_min, self.steer.grid.x_max),
            y_bounds=(self.steer.grid.y_min, self.steer.grid.y_max),
        )

        src_sampling = SourceSetSampler(
            target=[self.sources_mix],
            set=self.point_sources,
            replace=False,
            numsamples=3,
        )  # draw point sources from point_sources set (number of sources is sampled by nrcs_sampling object)
        if (self.max_nsources == self.min_nsources):
            src_sampling.numsamples = self.max_nsources

        rms_sampling = ContainerSampler(
            random_func=sample_rms)

        if not (self.max_nsources == self.min_nsources):  
            nsrc_sampling = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[src_sampling],
                attribute="numsamples",
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources),
            )
            return [mic_sampling, nsrc_sampling,src_sampling,rms_sampling, pos_sampling]
        else:
            return [mic_sampling, src_sampling,rms_sampling, pos_sampling]
                            
    def setup_input_features(self, cache_bf, cache_csm, cache_dir):

        # setup cache if necessary
        cache_dir = _handle_cache(cache_bf, cache_csm, cache_dir)
        self.freq_data.cached = cache_csm

        features = {} # dict with feature functions for pipeline object
        self._feature_objects = []
        fidx = self._get_freq_indices()
        if fidx is not None:
            # bound calculated frequencies for efficiency reasons
            self.freq_data.ind_low = min([f[0] for f in fidx])
            self.freq_data.ind_high = max([f[1] for f in fidx])    

        if "csm" in self.features:
            feature = CSMFeature(feature_name="csm",
                                 power_spectra=self.freq_data,
                                 fidx=fidx,
                                 cache_dir=cache_dir
                                 )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "csmtriu" in self.features:
            feature = NonRedundantCSMFeature(feature_name="csmtriu",
                                             power_spectra=self.freq_data,
                                             fidx=fidx,
                                             cache_dir=cache_dir
                                             )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "sourcemap" in self.features:
            self.beamformer.cached = cache_bf
            self.beamformer.freq_data = self.freq_data
            self.beamformer.steer = self.steer
            feature = SourceMapFeature(feature_name="sourcemap",
                                       beamformer=self.beamformer,
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=cache_dir
                                       )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "eigmode" in self.features:
            feature = EigmodeFeature(feature_name="eigmode",
                                    power_spectra=self.freq_data,
                                    fidx=fidx,
                                    cache_dir=cache_dir
                                     )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)
        
        return features

    def _setup_generation_process(self, tasks, head, log, logname):

        if tasks > 1:
            parallel=True
            ray.shutdown()
            ray.init(address=head)
        else:
            parallel = False

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(logname.split(".")[:-1]) + ".log")
        return parallel

    def generate(self, split, startsample=1, tasks=1, progress_bar=True, head=None, cache_csm=False, cache_bf=False, cache_dir=".", log=False):
        
        # Logging for debugging and timing statistic purpose
        logname = f"logfile_{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}" + ".log"
        # setup process
        parallel = self._setup_generation_process(tasks, head, log, logname)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample,
                           self.size, split)
        # yield the data
        for data in pipeline.get_data(progress_bar=progress_bar):
            yield data

    def save_tfrecord(self, split, name, startsample=1, tasks=1, progress_bar=True, head=None, cache_csm=False, cache_bf=False, cache_dir=".", log=False):
        if not TF_FLAG:
            raise ImportError("save data to .tfrecord format requires TensorFlow!")
        # setup process
        parallel = self._setup_generation_process(tasks, head, log, name)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        pipeline.progress_bar = progress_bar
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample,
                           self.size, split)
        
        # create Writer pipeline
        encoder_dict = {
            "loc": float_list_feature,
            "p2": float_list_feature,
            "idx": int64_feature,
            "seeds": int_list_feature,
        }
        for feature in self._feature_objects:
            feature.add_encoder_funcs(encoder_dict)
        # create TFRecordWriter to save pipeline output to TFRecord File
        WriteTFRecord(name=name, source=pipeline,
                      encoder_funcs=encoder_dict).save(progress_bar)

    def save_h5(self, split, name, startsample=1, tasks=1, progress_bar=True, head=None, cache_csm=False, cache_bf=False, cache_dir=".", log=False):
        # setup process
        parallel = self._setup_generation_process(tasks, head, log, name)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        pipeline.progress_bar = progress_bar
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample,
                           self.size, split)

        # create Writer pipeline
        metadata = {}
        for feature in self._feature_objects:
            metadata = feature.add_metadata(metadata.copy())

        WriteH5Dataset(name=name,
                       source=pipeline,
                       features=list(pipeline.features.keys()),
                       metadata=metadata.copy(),
                       ).save(progress_bar)  # start the calculation


    def get_feature_shapes(self):
        # number of frequencies
        fidx = self._get_freq_indices()
        if fidx is None:
            fdim = self.freq_data.fftfreq().shape[0]
        else: 
            fdim = len(fidx)
        # number of sources
        if self.max_nsources == self.min_nsources:
            ndim = self.max_nsources
        else:
            ndim = None
        # number of microphones
        mdim = self.steer.mics.num_mics
        features_shapes = {
            "idx" : (),
            "seeds" : (len(self.build_pipeline().sampler),),# TODO: this is not good...
            "loc" : (3,ndim),    
        }
        #gdim = self.steer.grid.shape
        if type(self).__name__ == "Dataset1":
            features_shapes.update({"p2" : (fdim,ndim)})
        elif type(self).__name__ == "Dataset2":
            features_shapes.update({"p2" : (fdim,ndim,ndim,2)})
        #for feature in self.features:
            #TODO: feature objects should know their own shape here!
        if "csm" in self.features:
            features_shapes.update({"csm" : (fdim,mdim,mdim,2)})
        if "csmtriu" in self.features:
            features_shapes.update({"csmtriu" : (fdim,mdim,mdim,1)})
        if "sourcemap" in self.features:
            features_shapes.update({"sourcemap" : (fdim,) + self.steer.grid.shape })
        return features_shapes


if TF_FLAG:
    import tensorflow as tf
    def get_tf_dataset(self, split, startsample=1, tasks=1, progress_bar=False, head=None, cache_csm=False, cache_bf=False, cache_dir=".", log=False): 
        signature = {k: tf.TensorSpec(shape,dtype=tf.float32,name=k) for k, shape in self.get_feature_shapes().items()}
        return tf.data.Dataset.from_generator(
            partial(
                self.generate,split, startsample,tasks,progress_bar,cache_csm,cache_bf,cache_dir,head,log
                ) ,output_signature=signature)                                   
    Dataset1.get_tf_dataset = get_tf_dataset
