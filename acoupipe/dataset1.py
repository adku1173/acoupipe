import ray
import numpy as np
from copy import deepcopy
from datetime import datetime
from os import path
from traits.api import HasPrivateTraits
from scipy.stats import poisson, norm
from acoular import MicGeom, WNoiseGenerator, PointSource, SourceMixer,\
    PowerSpectra, MaskedTimeInOut, Environment, RectGrid3D, BeamformerBase, BeamformerCleansc,\
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

dirpath = path.dirname(path.abspath(__file__))

config1 = {
    'version': "ds1-v01",  # data set version
    'max_nsources': 10,
    'min_nsources': 1,
    'c': 343.,  # speed of sound
    'fs': 40*343.,  # /ap with ap:1.0
    'blocksize': 128,  # block size used for FFT
    'overlap': '50%',
    'window': 'Hanning',
    'r_diag' : False,
    'T': 5,  # length of the simulated signal
    'micgeom': path.join(dirpath, "xml", "tub_vogel64_ap1.xml"),  # microphone positions
    'z_min' : 0.5,
    'z_max' : 0.5,
    'y_max' : .5,
    'y_min' : -.5,
    'x_max' : .5,
    'x_min' : -.5,
    'increment' : 1/63,
    'z' : 0.5,
    'ref_mic': 63,  # most center microphone
    'r': 0.05,  # integration radius for sector integration
    'steer_type': 'true level',
    'cache_csm': False,
    'cache_bf': False,
    'cache_dir': "./datasets",
}


class Dataset1:

    def __init__(self, split, size, features, f=None, num=0, startsample=1, config=config1):       
        self.split = split
        self.size = size
        self.startsample = startsample
        self.features = features
        self.f = f
        self.num = num
        self.config = config
        # private attributes(instances)
        self.mics = MicGeom(from_file=config['micgeom'])
        self.ap = self._get_aperture()
        self.noisy_mics = deepcopy(self.mics) # Microphone geometry with positional noise
        self.grid = RectGrid3D(
                x_min=config['x_min'], x_max=config['x_max'],
                y_min=config['y_min'], y_max=config['y_max'],
                z_min=config['z_min'], z_max=config['z_max'], increment=config['increment'])  # 64 x 64 grid
        self.env = Environment(c=config['c'])
        self.steer = SteeringVector(grid=self.grid, mics=self.mics, env=self.env, steer_type = config['steer_type'], 
                                    ref = self.mics.mpos[:, config['ref_mic']])
        # random variables
        self.random_var = {
            "mic_rvar" : norm(loc=0, scale=0.001), # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
            "loc_rvar" : norm(loc=0, scale=0.1688*self.ap),  # source positions
            "nsrc_rvar" : poisson(mu=3, loc=1)  # number of sources
        }

    def _get_aperture(self):
        max_ = 0
        for i in range(self.mics.num_mics):
            new_max = np.linalg.norm(self.mics.mpos_tot - self.mics.mpos_tot[:,i][:,np.newaxis],axis=0).max()
            max_ = max(max_,new_max)
        return max_

    def _fftfreq ( self ):
        return abs(np.fft.fftfreq(self.config['blocksize'], 1./self.config['fs'])\
                    [:int(self.config['blocksize']/2+1)])

    def _get_freq_indices(self):
        if self.f != None:
            if type(self.f) == float or type(self.f) == int:
                self.f = [self.f]
            fidx = [get_frequency_index_range(
                self._fftfreq(), f_, self.num) for f_ in self.f]
        else:
            fidx = None
        return fidx

    def build_pipeline(self, parallel=False):
        c = self.config
        # Assumed microphone geometry without positional noise
        white_noise_signals = [
            WNoiseGenerator(sample_freq=c['fs'], seed=i+1, numsamples=c['T']*c['fs']) for i in range(c['max_nsources'])
        ]  # Signals
        self.point_sources = [
            PointSource(signal=signal, mics=self.noisy_mics, env=self.env, loc=(0, 0, c['z'])) for signal in white_noise_signals
        ]  # Monopole sources emitting the white noise signals
        # Source Mixer mixing the signals of all sources (number will be sampled)
        self.sources_mix = SourceMixer(sources=self.point_sources)

        # Set up PowerSpectra objects to calculate CSM feature and reference p2 value
        # first object is used to calculate the full CSM
        # second object will be used to calculate the p2 value at the reference microphone (for each present source)
        ps_args = {'block_size': c['blocksize'], 'overlap': c['overlap'],
                   'window': c['window'], 'precision': 'complex64'}
        self.ps_csm = PowerSpectra(time_data=self.sources_mix,
                              cached=c['cache_csm'], **ps_args)
        # caching takes more time than calculation for a single channel
        ref_channel = MaskedTimeInOut(source=self.sources_mix, invalid_channels=[_ for _ in range(
            self.mics.num_mics) if not _ == c['ref_mic']])  # masking other channels than the reference channel
        ps_ref = PowerSpectra(time_data=ref_channel, cached=False, **ps_args)
    #    spectra_inout = SpectraInOut(source=self.sources_mix,block_size=blocksize, window="Hanning",overlap="50%" )
        
        # set up the feature dict with methods to get the labels
        fidx = self._get_freq_indices()
        features = {  # (callable, arg1, arg2, ...)
            "loc": (lambda smix: np.array([s.loc for s in smix.sources], dtype=np.float32).T, self.sources_mix),
            "p2": (get_source_p2, self.sources_mix, ps_ref, fidx, None),
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
            x_bounds=(self.config['x_min'], self.config['x_max']),
            y_bounds=(self.config['y_min'], self.config['y_max']),
        )

        src_sampling = SourceSetSampler(
            target=[self.sources_mix],
            set=self.point_sources,
            replace=False,
            numsamples=3,
        )  # draw point sources from point_sources set (number of sources is sampled by nrcs_sampling object)
        if (self.config['max_nsources'] == self.config['min_nsources']):
            src_sampling.numsamples = self.config['max_nsources']

        rms_sampling = ContainerSampler(
            random_func=sample_rms)

        if not (self.config['max_nsources'] == self.config['min_nsources']):  
            nsrc_sampling = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[src_sampling],
                attribute='numsamples',
                filter=lambda x: (x <= self.config['max_nsources']) and (
                    x >= self.config['min_nsources']),
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
            self.ps_csm.ind_low = min([f[0] for f in fidx])
            self.ps_csm.ind_high = max([f[1] for f in fidx])    

        # handle cache
        cache_dir = _handle_cache(
            self.config['cache_bf'], self.config['cache_csm'], self.config['cache_dir'])
        bb_args = {'r_diag': self.config['r_diag'], 'cached': self.config['cache_bf'], 'precision': 'float32'}                   

        if "csm" in self.features:
            feature = CSMFeature(feature_name="csm",
                                 power_spectra=self.ps_csm,
                                 fidx=fidx,
                                 cache_dir=cache_dir
                                 )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "csmtriu" in self.features:
            feature = NonRedundantCSMFeature(feature_name="csmtriu",
                                             power_spectra=self.ps_csm,
                                             fidx=fidx,
                                             cache_dir=cache_dir
                                             )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "sourcemap" in self.features:
            feature = SourceMapFeature(feature_name="sourcemap",
                                       beamformer=BeamformerBase(freq_data=self.ps_csm, steer=self.steer, **bb_args),
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=cache_dir
                                       )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        if "cleansc" in self.features:
            feature = SourceMapFeature(feature_name="cleansc",
                                       beamformer=BeamformerCleansc(freq_data=self.ps_csm, steer=self.steer, **bb_args),
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=cache_dir
                                       )
            features = feature.add_feature_funcs(features)
            self._feature_objects.append(feature)

        # if "ref_cleansc" in self.features:
        #     feature = RefSourceMapFeature(feature_name="ref_cleansc",
        #                                   beamformer=BeamformerCleansc(freq_data=self.ps_csm, steer=self.steer, **bb_args),
        #                                   sourcemixer=self.sources_mix,
        #                                   powerspectra=ps_ref,
        #                                   r=config['r'],
        #                                   f=self.f,
        #                                   num=self.num,
        #                                   cache_dir=cache_dir
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
            "nsources": int64_feature,
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
        metadata = self.config.copy()
        for feature in self._feature_objects:
            metadata = feature.add_metadata(metadata.copy())

        WriteH5Dataset(name=name,
                       source=pipeline,
                       features=list(pipeline.features.keys()),
                       metadata=metadata.copy(),
                       ).save()  # start the calculation


    
