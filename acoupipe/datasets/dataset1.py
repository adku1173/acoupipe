import ray
import numpy as np
from datetime import datetime
from os import path
from scipy.stats import poisson, norm
from acoular import config, MicGeom, WNoiseGenerator, PointSource, SourceMixer,\
    PowerSpectra, MaskedTimeInOut, Environment, RectGrid, BeamformerBase, BeamformerCleansc,\
    SteeringVector
from acoupipe import MicGeomSampler, PointSourceSampler, SourceSetSampler, \
    NumericAttributeSampler, ContainerSampler, DistributedPipeline, BasePipeline,\
    WriteH5Dataset, WriteTFRecord
from .features import RefSourceMapFeature, get_source_p2, CSMFeature, SourceMapFeature,\
    NonRedundantCSMFeature
from .helper import set_pipeline_seeds, get_frequency_index_range, _handle_cache, _handle_log

TF_FLAG = True
try:
    from acoupipe import float_list_feature, int64_feature, int_list_feature
except:
    TF_FLAG = False

dirpath = path.dirname(path.abspath(__file__))
mg = MicGeom(from_file=path.join(dirpath, "tub_vogel64_ap1.xml"))

constants = {
    'MAX_NSOURCES': 10,
    'MIN_NSOURCES': 1,
    'AP': 1.,
    'VERSION': "ds1-v001",  # data set version
    'C': 343.,  # speed of sound
    'HE': 40,  # Helmholtz number (defines the sampling frequency)
    'SFREQ': 40*343./1.,  # /ap with ap:1.0
    'BLOCKSIZE': 128,  # block size used for FFT
    'OVERLAP': '50%',
    'WINDOW': 'Hanning',
    'SIGLENGTH': 5,  # length of the simulated signal
    'MGEOM': mg.mpos_tot.copy(),  # microphone positions
    'REF_MIC': 63,  # index of the reference microphone
    'R': 0.05,  # integration radius for sector integration
    'CACHE_CSM': False,
    'CACHE_BF': False,
    'CACHE_DIR': "./datasets",
}


class Dataset:

    def __init__(self, split, numsamples, features, f=None, num=0, startsample=1, constants=constants, tasks=1, head=None):
        self.split = split
        self.numsamples = numsamples
        self.startsample = startsample
        self.features = features
        self.f = f
        self.num = num
        self.constants = constants
        self.tasks = tasks
        self.head = head

    def build_pipeline(self):
        c = self.constants
        if c['MAX_NSOURCES'] == c['MIN_NSOURCES']:
            nsources_constant = True
        else:
            nsources_constant = False

        # handle cache
        cache_dir = _handle_cache(
            c['CACHE_BF'], c['CACHE_CSM'], c['CACHE_DIR'])

        # Random Variables
        # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
        mic_rvar = norm(loc=0, scale=0.001*c['AP'])
        pos_rvar = norm(loc=0, scale=0.1688*c['AP'])  # source positions
        nsrc_rvar = poisson(mu=3, loc=1)  # number of sources

        # Computational Pipeline Acoular
        # Microphone geometry with positional noise
        mg_manipulated = MicGeom(mpos_tot=c['MGEOM'].copy())
        # Assumed microphone geometry without positional noise
        mg_fixed = MicGeom(mpos_tot=c['MGEOM'].copy())
        env = Environment(c=c['C'], roi=np.array([]))  # Environment
        white_noise_signals = [
            WNoiseGenerator(sample_freq=c['SFREQ'], seed=i+1, numsamples=c['SIGLENGTH']*c['SFREQ']) for i in range(c['MAX_NSOURCES'])
        ]  # Signals
        point_sources = [
            PointSource(signal=signal, mics=mg_manipulated, env=env, loc=(0, 0, .5*c['AP'])) for signal in white_noise_signals
        ]  # Monopole sources emitting the white noise signals
        # Source Mixer mixing the signals of all sources (number will be sampled)
        sources_mix = SourceMixer(sources=point_sources)

        # Set up PowerSpectra objects to calculate CSM feature and reference p2 value
        # first object is used to calculate the full CSM
        # second object will be used to calculate the p2 value at the reference microphone (for each present source)
        ps_args = {'block_size': c['BLOCKSIZE'], 'overlap': c['OVERLAP'],
                   'window': c['WINDOW'], 'precision': 'complex64'}
        ps_csm = PowerSpectra(time_data=sources_mix,
                              cached=c['CACHE_CSM'], **ps_args)
        # caching takes more time than calculation for a single channel
        ref_channel = MaskedTimeInOut(source=sources_mix, invalid_channels=[_ for _ in range(
            mg_fixed.num_mics) if not _ == c['REF_MIC']])  # masking other channels than the reference channel
        ps_ref = PowerSpectra(time_data=ref_channel, cached=False, **ps_args)
    #    spectra_inout = SpectraInOut(source=sources_mix,block_size=BLOCKSIZE, window="Hanning",overlap="50%" )

        # Set up Beamformer object to calculate sourcemap feature
        if ("sourcemap" in self.features) or ("ref_cleansc" in self.features):
            bb_args = {'r_diag': False, }
            sv_args = {'steer_type': 'true level',
                       'ref': mg_fixed.mpos[:, c['REF_MIC']]}
            rg = RectGrid(
                x_min=-0.5*c['AP'], x_max=0.5*c['AP'],
                y_min=-0.5*c['AP'], y_max=0.5*c['AP'],
                z=.5*c['AP'], increment=1/63)  # 64 x 64 grid
            st = SteeringVector(
                grid=rg, mics=mg_fixed, env=env, **sv_args)
            bb = BeamformerBase(
                freq_data=ps_csm, steer=st, cached=c['CACHE_BF'], precision='float32', **bb_args)
            bfcleansc = BeamformerCleansc(
                freq_data=ps_csm, steer=st, cached=c['CACHE_BF'], precision='float32', **bb_args)

        # Computational Pipeline AcouPipe
        # callable function to draw and assign sound pressure RMS values to the sources of the SourceMixer object
        def sample_rms(rng):
            "draw source pressures square, Rayleigh distribution, sort them, calc rms"
            nsrc = len(sources_mix.sources)
            # draw source pressures square, Rayleigh distribution, sort them, calc rms
            p_rms = np.sqrt(np.sort(rng.rayleigh(5, nsrc))[::-1])
            p_rms /= p_rms.max()  # norm it
            for i, rms in enumerate(p_rms):
                sources_mix.sources[i].signal.rms = rms  # set rms value

        mic_sampling = MicGeomSampler(
            random_var=mic_rvar,
            target=mg_manipulated,
            ddir=np.array([[1.0], [1.0], [0]])
        )  # ddir along two dimensions -> bivariate sampling

        pos_sampling = PointSourceSampler(
            random_var=pos_rvar,
            target=sources_mix.sources,
            # ldir: 1.0 along first two dimensions -> bivariate sampling
            ldir=np.array([[1.0], [1.0], [0.0]]),
            # only allow values between -.5 and .5
            x_bounds=(-.5*c['AP'], .5*c['AP']),
            y_bounds=(-.5*c['AP'], .5*c['AP']),
        )

        src_sampling = SourceSetSampler(
            target=[sources_mix],
            set=point_sources,
            replace=False,
            numsamples=3,
        )  # draw point sources from point_sources set (number of sources is sampled by nrcs_sampling object)
        if nsources_constant:
            src_sampling.numsamples = c['MAX_NSOURCES']

        rms_sampling = ContainerSampler(
            random_func=sample_rms)

        if not nsources_constant:  # if no number of sources is specified, the number of sources will be samples randomly
            nsrc_sampling = NumericAttributeSampler(
                random_var=nsrc_rvar,
                target=[src_sampling],
                attribute='numsamples',
                filter=lambda x: (x <= c['MAX_NSOURCES']) and (
                    x >= c['MIN_NSOURCES']),
            )

            sampler_list = [mic_sampling, nsrc_sampling,
                            src_sampling,
                            rms_sampling, pos_sampling]
        else:
            sampler_list = [mic_sampling,
                            src_sampling,
                            rms_sampling, pos_sampling]

        if self.tasks > 1:
            pipeline = DistributedPipeline(
                sampler=sampler_list,
                numworkers=self.tasks,
            )
        else:
            pipeline = BasePipeline(
                sampler=sampler_list,
            )

        # desired frequency
        if self.f != None:
            if type(self.f) == float or type(self.f) == int:
                f = [f]
            fidx = [get_frequency_index_range(
                ps_csm.fftfreq(), f_, self.num) for f_ in self.f]
            ps_csm.ind_low = min([f[0] for f in fidx])
            ps_csm.ind_high = max([f[1] for f in fidx])
        else:
            fidx = None
            f = None
        # set up the feature dict with methods to get the labels
        feature_methods = {  # (callable, arg1, arg2, ...)
            "loc": (lambda smix: np.array([s.loc for s in smix.sources], dtype=np.float32).T, sources_mix),
            "nsources": (lambda smix: len(smix.sources), sources_mix),
            "p2": (get_source_p2, sources_mix, ps_ref, fidx, cache_dir),
        }

        feature_objects = []
        if "csm" in self.features:
            feature = CSMFeature(feature_name="csm",
                                 power_spectra=ps_csm,
                                 fidx=fidx,
                                 cache_dir=cache_dir
                                 )
            feature_methods = feature.add_feature_funcs(feature_methods)
            feature_objects.append(feature)

        if "csmtriu" in self.features:
            feature = NonRedundantCSMFeature(feature_name="csmtriu",
                                             power_spectra=ps_csm,
                                             fidx=fidx,
                                             cache_dir=cache_dir
                                             )
            feature_methods = feature.add_feature_funcs(feature_methods)
            feature_objects.append(feature)

        if "sourcemap" in self.features:
            feature = SourceMapFeature(feature_name="sourcemap",
                                       beamformer=bb,
                                       f=self.f,
                                       num=self.num,
                                       cache_dir=cache_dir
                                       )
            feature_methods = feature.add_feature_funcs(feature_methods)
            feature_objects.append(feature)

        if "ref_cleansc" in self.features:
            feature = RefSourceMapFeature(feature_name="ref_cleansc",
                                          beamformer=bfcleansc,
                                          sourcemixer=sources_mix,
                                          powerspectra=ps_ref,
                                          r=c['R'],
                                          f=self.f,
                                          num=self.num,
                                          cache_dir=cache_dir
                                          )
            feature_methods = feature.add_feature_funcs(feature_methods)
            feature_objects.append(feature)

        # add features to the pipeline
        pipeline.features = feature_methods
        self._feature_objects = feature_objects
        return pipeline

    def simulate(self, log=False):
        # Ray Config
        if self.tasks > 1:
            ray.shutdown()
            ray.init(address=self.head)

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(
                fname=f"logfile_{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}" + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline()
        # set seeds
        set_pipeline_seeds(pipeline, self.startsample,
                           self.numsamples, self.split)

        # yield the data
        for data in pipeline.get_data():
            yield data

    def save_tfrecord(self, name, log=False):
        if not TF_FLAG:
            raise ImportError("save data to .tfrecord format requires TensorFlow!")

        # Ray Config
        if self.tasks > 1:
            ray.shutdown()
            ray.init(address=self.head)

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(name.split('.')[:-1]) + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline()
        # set seeds
        set_pipeline_seeds(pipeline, self.startsample,
                           self.numsamples, self.split)
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

    def save_h5(self, name, log=False):
        if self.tasks > 1:
            ray.shutdown()
            ray.init(address=self.head)

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(name.split('.')[:-1]) + ".log")

        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline()
        # set seeds
        set_pipeline_seeds(pipeline, self.startsample,
                           self.numsamples, self.split)

        # create Writer pipeline
        metadata = self.constants.copy()
        for feature in self._feature_objects:
            metadata = feature.add_metadata(metadata.copy())

        WriteH5Dataset(name=name,
                       source=pipeline,
                       features=list(pipeline.features.keys()),
                       metadata=metadata.copy(),
                       ).save()  # start the calculation
