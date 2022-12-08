import ray
import numpy as np
from copy import deepcopy
from datetime import datetime
from os import path
from traits.api import HasPrivateTraits
from scipy.stats import poisson, norm
from acoular import MicGeom, WNoiseGenerator, PointSource, SourceMixer,\
    PowerSpectra, MaskedTimeInOut, Environment, RectGrid, BeamformerBase, BeamformerCleansc,\
    SteeringVector
from .sampler import MicGeomSampler, PointSourceSampler, SourceSetSampler, \
    NumericAttributeSampler, ContainerSampler
from .pipeline import DistributedPipeline, BasePipeline
from .writer import WriteH5Dataset
from .features import RefSourceMapFeature, get_source_p2, CSMFeature, SourceMapFeature,\
    NonRedundantCSMFeature
from .helper import set_pipeline_seeds, get_frequency_index_range, _handle_cache, _handle_log
from .config import TF_FLAG
if TF_FLAG:
    from .writer import float_list_feature, int64_feature, int_list_feature, WriteTFRecord

VERSION = "ds1-v01"
dirpath = path.dirname(path.abspath(__file__))

class Dataset1:

    def __init__(
            self, 
            split, 
            size, 
            features, 
            f=None, 
            num=0, 
            fs=40*343.,
            startsample=1, 
            max_nsources = 10,
            min_nsources = 1,
            env = Environment(c=343.),
            mics = MicGeom(from_file=path.join(dirpath, "xml", "tub_vogel64_ap1.xml")),
            grid = RectGrid(y_min=-.5,y_max=.5,x_min=-.5,x_max=.5,z=.5,increment=1/63),
            cache_csm = False,
            cache_bf = False,
            cache_dir = "./datasets",
            config=None):       
        self.split = split
        self.size = size
        self.startsample = startsample
        self.features = features
        self.f = f
        self.num = num
        self.fs = fs
        self.max_nsources = max_nsources
        self.min_nsources = min_nsources
        self.env = env
        self.mics = mics 
        self.grid = grid
        self.cache_csm = cache_csm
        self.cache_bf = cache_bf
        self.cache_dir = _handle_cache(cache_bf, cache_csm, cache_dir)
        self.config = config
       # dependent attributes / objects
        self.ref_mic = np.argmin(np.linalg.norm((mics.mpos - mics.center[:,np.newaxis]),axis=0))
        self.steer = SteeringVector(
            grid=self.grid, mics=self.mics, env=self.env, steer_type ='true level', ref = self.mics.mpos[:,self.ref_mic])
        self.freq_data = PowerSpectra(time_data=PointSource(signal=WNoiseGenerator(sample_freq=self.fs)),
            block_size=128, overlap="50%", window='Hanning', precision='complex64', cached=self.cache_csm)
        # random variables
        self.random_var = {
            "mic_rvar" : norm(loc=0, scale=0.001), # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
            "loc_rvar" : norm(loc=0, scale=0.1688*self.mics.aperture),  # source positions
            "nsrc_rvar" : poisson(mu=3, loc=1)  # number of sources
        }

    def _get_freq_indices(self):
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                self.f = [self.f]
            fidx = [get_frequency_index_range(
                self.freq_data.fftfreq(), f_, self.num) for f_ in self.f]
        else:
            fidx = None
        return fidx

    def build_pipeline(self, parallel=False):
        # create copy for noisy positions
        self.noisy_mics = deepcopy(self.mics) # Microphone geometry with positional noise
        white_noise_signals = [ # create source signal array
            WNoiseGenerator(sample_freq=self.fs, seed=i+1, numsamples=5*self.fs) for i in range(self.max_nsources)
        ]  # Signals
        self.point_sources = [
            PointSource(signal=signal, mics=self.noisy_mics, env=self.env, loc=(0, 0, self.grid.z)) for signal in white_noise_signals
        ]  # Monopole sources emitting the white noise signals
        self.sources_mix = SourceMixer(sources=self.point_sources) # Source Mixer mixing the signals of all sources (number will be sampled)
        self.freq_data.time_data = self.sources_mix
        self.source_freq_data = deepcopy(self.freq_data) # will be used to calculate the p2 value at the reference microphone (for each present source)
        self.source_freq_data.cached = False
        self.source_freq_data.time_data = MaskedTimeInOut(source=self.sources_mix, invalid_channels=[_ for _ in range(
            self.mics.num_mics) if not _ == self.ref_mic]) # mask all channels except ref mic       
        # set up the feature dict with methods to get the labels
        fidx = self._get_freq_indices()
        features = {  # (callable, arg1, arg2, ...)
            "loc": (lambda smix: np.array([s.loc for s in smix.sources], dtype=np.float32).T, self.sources_mix),
            "p2": (get_source_p2, self.sources_mix, self.source_freq_data, fidx, None),
        }
        # add input features (csm, sourcemap, cleansc, ...)
        features.update(self.setup_features())
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=self.setup_sampler(),features=features)

    def setup_sampler(self):
        # callable function to draw and assign sound pressure RMS values to the sources of the SourceMixer object
        def sample_rms(rng):
            "draw source pressures square, Rayleigh distribution, sort them, calc rms"
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
            x_bounds=(self.grid.x_min, self.grid.x_max),
            y_bounds=(self.grid.y_min, self.grid.y_max),
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
                attribute='numsamples',
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources),
            )
            return [mic_sampling, nsrc_sampling,src_sampling,rms_sampling, pos_sampling]
        else:
            return [mic_sampling, src_sampling,rms_sampling, pos_sampling]
                            
    def setup_features(self):
        features = {} # dict with feature functions for pipeline object
        self._feature_objects = []
        fidx = self._get_freq_indices()
        if not fidx is None:
            # bound calculated frequencies for efficiency reasons
            self.freq_data.ind_low = min([f[0] for f in fidx])
            self.freq_data.ind_high = max([f[1] for f in fidx])    

        if "csm" in self.features:
            feature = CSMFeature(feature_name="csm",
                                 power_spectra=self.freq_data,
                                 fidx=fidx,
                                 cache_dir=self.cache_dir
                                 )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "csmtriu" in self.features:
            feature = NonRedundantCSMFeature(feature_name="csmtriu",
                                             power_spectra=self.freq_data,
                                             fidx=fidx,
                                             cache_dir=self.cache_dir
                                             )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        bb_args = {'r_diag': False, 'cached': self.cache_bf, 'precision': 'float32'}                   
        if "sourcemap" in self.features:
            feature = SourceMapFeature(feature_name="sourcemap",
                                       beamformer=BeamformerBase(freq_data=self.freq_data, steer=self.steer, **bb_args),
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=self.cache_dir
                                       )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "cleansc" in self.features:
            feature = SourceMapFeature(feature_name="cleansc",
                                       beamformer=BeamformerCleansc(freq_data=self.freq_data, steer=self.steer, **bb_args),
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=self.cache_dir
                                       )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        # if "ref_cleansc" in self.features:
        #     feature = RefSourceMapFeature(feature_name="ref_cleansc",
        #                                   beamformer=BeamformerCleansc(freq_data=self.freq_data, steer=self.steer, **bb_args),
        #                                   sourcemixer=self.sources_mix,
        #                                   powerspectra=ps_ref,
        #                                   r=0.05,
        #                                   f=self.f,
        #                                   num=self.num,
        #                                   cache_dir=self.cache_dir
        #                                   )
        #     features = feature.add_feature_funcs(features)
        #     self._feature_objects.append(feature)
        return features

    def generate(self, tasks=1, head=None, log=False):
        # Ray Config
        if tasks > 1:
            parallel=True
            ray.shutdown()
            ray.init(address=head)
        else:
            parallel = False
        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(
                fname=f"logfile_{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}" + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, self.startsample,
                           self.size, self.split)
        # yield the data
        for data in pipeline.get_data():
            yield data

    def save_tfrecord(self, name, tasks=1, head=None, log=False):
        if not TF_FLAG:
            raise ImportError("save data to .tfrecord format requires TensorFlow!")

        # Ray Config
        if tasks > 1:
            parallel=True
            ray.shutdown()
            ray.init(address=head)
        else:
            parallel = False
        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(name.split('.')[:-1]) + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, self.startsample,
                           self.size, self.split)
        
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
                      encoder_funcs=encoder_dict).save()

    def save_h5(self, name, tasks=1, head=None, log=False):
        if tasks > 1:
            parallel=True
            ray.shutdown()
            ray.init(address=head)
        else:
            parallel = False

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(name.split('.')[:-1]) + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, self.startsample,
                           self.size, self.split)

        # create Writer pipeline
        metadata = {}
        for feature in self._feature_objects:
            metadata = feature.add_metadata(metadata.copy())

        WriteH5Dataset(name=name,
                       source=pipeline,
                       features=list(pipeline.features.keys()),
                       metadata=metadata.copy(),
                       ).save()  # start the calculation


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
        mdim = self.mics.num_mics
        features_shapes = {
            "idx" : (),
            "seeds" : (len(self.build_pipeline().sampler),),# TODO: this is not good...
            "loc" : (3,ndim),    
        }
        gdim = self.grid.shape
        if type(self).__name__ == 'Dataset1':
            features_shapes.update({"p2" : (fdim,ndim)})
        elif type(self).__name__ == 'Dataset2':
            features_shapes.update({"p2" : (fdim,ndim,ndim,2)})
        #for feature in self.features:
            #TODO: feature objects should know their own shape here!
        if "csm" in self.features:
            features_shapes.update({"csm" : (fdim,mdim,mdim,2)})
        if "csmtriu" in self.features:
            features_shapes.update({"csmtriu" : (fdim,mdim,mdim,1)})
        if "sourcemap" in self.features:
            features_shapes.update({"sourcemap" : (fdim,) + self.grid.shape })
        if "cleansc" in self.features:
            features_shapes.update({"cleansc" : (fdim,) + self.grid.shape })
        return features_shapes

from acoupipe import TF_FLAG
if TF_FLAG:
    import tensorflow as tf
    def get_tf_dataset(self):
        signature = {k: tf.TensorSpec(shape,dtype=tf.float32,name=k) for k, shape in self.get_feature_shapes().items()}
        return tf.data.Dataset.from_generator(self.generate,output_signature=signature)                                   
    setattr(Dataset1,'get_tf_dataset',get_tf_dataset)
