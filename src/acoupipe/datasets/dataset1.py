from copy import deepcopy
from datetime import datetime
from functools import partial

import ray
import scipy as sc
from acoular import (
    BeamformerBase,
    Environment,
    FiltWNoiseGenerator,
    MaskedTimeInOut,
    MicGeom,
    PointSource,
    PowerSpectra,
    RectGrid,
    SamplesGenerator,
    SourceMixer,
    SteeringVector,
    WNoiseGenerator,
)
from numpy import argmin, array, diagonal, float32, linalg, newaxis, real, sort, sqrt
from scipy.stats import norm, poisson, rayleigh

from acoupipe.config import TF_FLAG
from acoupipe.datasets.features import get_csm, get_eigmode, get_nonredundant_csm, get_source_p2, get_sourcemap
from acoupipe.datasets.helper import _handle_cache, _handle_log, get_frequency_index_range, set_pipeline_seeds
from acoupipe.datasets.micgeom import tub_vogel64_ap1
from acoupipe.filter import generate_uniform_parametric_eq
from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.sampler import ContainerSampler, CovSampler, MicGeomSampler, NumericAttributeSampler, PointSourceSampler, SourceSetSampler
from acoupipe.writer import WriteH5Dataset

VERSION = "ds1-v02"
DEFAULT_ENV = Environment(c=343.)
DEFAULT_MICS = MicGeom(mpos_tot=tub_vogel64_ap1)
ref_mic_idx = argmin(linalg.norm((DEFAULT_MICS.mpos - DEFAULT_MICS.center[:,newaxis]),axis=0))
DEFAULT_GRID = RectGrid(y_min=-.5,y_max=.5,x_min=-.5,x_max=.5,z=.5,increment=1/63)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")
DEFAULT_STEER = SteeringVector(grid=DEFAULT_GRID, mics=DEFAULT_MICS, env=DEFAULT_ENV, steer_type ="true level",
                                ref=DEFAULT_MICS.mpos[:,ref_mic_idx])
DEFAULT_FREQ_DATA = PowerSpectra(time_data=SamplesGenerator(sample_freq=40*343),
                            block_size=128, overlap="50%", window="Hanning", precision="complex64")
DEFAULT_RANDOM_VAR = {
            "mic_rvar" : norm(loc=0, scale=0.001), # microphone array position noise; std -> 0.001 = 0.1% of the aperture size
            "p2_rvar" : rayleigh(5),
            "loc_rvar" : norm(loc=0, scale=0.1688*DEFAULT_MICS.aperture),  # source positions
            "nsrc_rvar" : poisson(mu=3, loc=1),  # number of sources
        }

class Dataset1:

    def __init__(
            self,
            features,
            f=None,
            num=0,
            fs=40*343.,
            max_nsources = 10,
            min_nsources = 1,
            steer = DEFAULT_STEER,
            beamformer = DEFAULT_BEAMFORMER,
            freq_data = DEFAULT_FREQ_DATA,
            random_var = DEFAULT_RANDOM_VAR,
            sample_spectra=False,
            ):
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
        self.sample_spectra = sample_spectra
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                self.f = [self.f]

    def _get_freq_indices(self):
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                self.f = [self.f]
            fidx = [get_frequency_index_range(
                self.freq_data.fftfreq(), f_, self.num) for f_ in self.f]
        else:
            fidx = None
        return fidx

    def get_dataset_metadata(self):
        metadata = {}
        metadata["features"] = "-".join(self.features)
        if self.f is not None:
            metadata["f"] = "-".join(map(str,self.f))
        else:
            metadata["f"] = "all"
        metadata["num"] = self.num
        metadata["fs"] = self.fs
        metadata["max_nsources"] = self.max_nsources
        metadata["min_nsources"] = self.min_nsources
        metadata["version"] = VERSION
        return metadata

    def get_dataset_feature_names(self):
        feature_names = ["idx","seeds","p2", "loc"]
        if "csm" in self.features:
            feature_names.append("csm")
        if "csmtriu" in self.features:
            feature_names.append("csmtriu")
        if "eigmode" in self.features:
            feature_names.append("eigmode")
        if "sourcemap" in self.features:
            feature_names.append("sourcemap")
        return feature_names

    def build_sampler(self):
        noisy_mics = deepcopy(self.steer.mics) # Microphone geometry with positional noise
        if self.sample_spectra:
            Signal = FiltWNoiseGenerator
        else:
            Signal = WNoiseGenerator
        noise_signals = [
            Signal(sample_freq=self.fs, seed=i+1, numsamples=5*self.fs) for i in range(self.max_nsources)
        ]
        point_sources = [PointSource(
                signal=signal, mics=noisy_mics, env=self.steer.env, loc=(0, 0, self.steer.grid.z))
                for signal in noise_signals]  # Monopole sources emitting the white noise signals
        sourcemixer = SourceMixer(sources=point_sources) # Source Mixer mixing the signals of all sources
        sampler = {}
        sampler[1] = MicGeomSampler(
            random_var=self.random_var["mic_rvar"],
            target=noisy_mics,
            ddir=array([[1.0], [1.0], [0]])
        )  # ddir along two dimensions -> bivariate sampling

        srcsampler = SourceSetSampler(
            target=[sourcemixer],
            set=point_sources,
            replace=False,
            nsources=self.max_nsources, #  number of sources is sampled by nrcs_sampling object
        )  # draw point sources from point_sources set
        sampler[2] = srcsampler

        strength_sampler = CovSampler(
            random_var = self.random_var["p2_rvar"],
            nsources = self.max_nsources,
            scale_variance = False,
            nfft = 1)
        sampler[3] = strength_sampler

        loc_sampler = PointSourceSampler(
            random_var=self.random_var["loc_rvar"],
            target=sourcemixer.sources,
            # ldir:1.0 along first two dimensions -> bivariate sampling
            ldir=array([[1.0], [1.0], [0.0]]),
            x_bounds=(self.steer.grid.x_min, self.steer.grid.x_max),
            y_bounds=(self.steer.grid.y_min, self.steer.grid.y_max),)
        sampler[4] = loc_sampler

        if not (self.max_nsources == self.min_nsources):
            sampler[0] = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[strength_sampler, srcsampler],
                attribute="nsources",
                single_value = True,
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources))

        if self.sample_spectra:
            def sample_spectra(signals, rng):
                for s in signals:
                    if isinstance(s, FiltWNoiseGenerator):
                        _, sos = generate_uniform_parametric_eq(
                            self.freq_data.block_size//2+1, 16, rng=rng)
                        b, a = sc.signal.zpk2tf(*sc.signal.sos2zpk(sos))
                        s.ar = b
                        s.ma = a
                return
            sampler[5] = ContainerSampler(
                random_func = partial(sample_spectra, noise_signals),
            )

        return sampler

    def build_pipeline(self, parallel, cache_csm, cache_bf, cache_dir):
        cache_dir = _handle_cache(cache_bf, cache_csm, cache_dir)
        ref_mic_idx = argmin(linalg.norm((self.steer.mics.mpos - self.steer.mics.center[:,newaxis]),axis=0))
        self.freq_data.cached = cache_csm
        self.beamformer.steer = deepcopy(self.steer)
        self.beamformer.cached = cache_bf
        # set up sampler
        sampler = self.build_sampler()
        # set up the feature dict with methods to get the labels
        fidx = self._get_freq_indices()
        if fidx is not None:
            # bound calculated frequencies for efficiency reasons
            self.freq_data.ind_low = min([f[0] for f in fidx])
            self.freq_data.ind_high = max([f[1] for f in fidx])
        else:
            self.freq_data.ind_low = 0
            self.freq_data.ind_high = None
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler,
                        features=partial(calc_features,
                                freq_data=self.freq_data,
                                beamformer=self.beamformer,
                                input_features=self.features,
                                fidx=fidx,
                                f=self.f,
                                num=self.num,
                                cache_bf = cache_bf,
                                cache_csm = cache_csm,
                                cache_dir = cache_dir,
                                ref_mic_idx=ref_mic_idx))


    def _setup_generation_process(self, tasks, address, log, logname):

        if tasks > 1:
            parallel=True
            ray.shutdown()
            ray.init(address=address)
        else:
            parallel = False

        # Logging for debugging and timing statistic purpose
        if log:
            _handle_log(".".join(logname.split(".")[:-1]) + ".log")
        return parallel

    def generate(self, split, size, startsample=1, tasks=1, progress_bar=True, address=None, cache_csm=False, cache_bf=False,
                cache_dir=".", log=False):

        # Logging for debugging and timing statistic purpose
        logname = f"logfile_{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}" + ".log"
        # setup process
        parallel = self._setup_generation_process(tasks, address, log, logname)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample, size, split)

        # yield the data
        for data in pipeline.get_data(progress_bar=progress_bar):
            yield data

    def save_h5(self, split, size, name, startsample=1, tasks=1, progress_bar=True, address=None, cache_csm=False, cache_bf=False,
                cache_dir=".", log=False):
        # setup process
        parallel = self._setup_generation_process(tasks, address, log, name)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample, size, split)

        # create Writer pipeline
        WriteH5Dataset(name=name,
                       source=pipeline,
                       metadata=self.get_dataset_metadata(),
                       ).save(progress_bar)  # start the calculation

    def get_feature_shapes(self):
        # number of samplers
        sampler = self.build_sampler()
        sdim = len(sampler.values())
        del sampler
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
            "seeds" : (sdim,2),
            "loc" : (3,ndim),
            "p2" : (fdim,ndim)}
        if "csm" in self.features:
            features_shapes.update({"csm" : (fdim,mdim,mdim,2)})
        if "csmtriu" in self.features:
            features_shapes.update({"csmtriu" : (fdim,mdim,mdim,1)})
        if "sourcemap" in self.features:
            features_shapes.update({"sourcemap" : (fdim,) + self.steer.grid.shape })
        if "eigmode" in self.features:
            features_shapes.update({"eigmode" : (fdim,mdim,mdim,2) })
        return features_shapes


def calc_features(s, freq_data, beamformer, input_features, fidx, f, num, cache_bf, cache_csm, cache_dir, ref_mic_idx):
    sourcemixer = s[2].target[0]
    # apply amplitudes
    prms = sqrt(real(sort(diagonal(s[3].target[0]))))
    for i,src in enumerate(sourcemixer.sources):
        src.signal.rms = prms[i]
    freq_data.time_data = sourcemixer# the sampled source mixer
    beamformer.freq_data = freq_data # change the freq_data, but not the steering!
    # the measured p2
    source_freq_data = deepcopy(freq_data) # will be used to calculate the p2 value at the reference microphone
    source_freq_data.cached = False
    source_freq_data.time_data = MaskedTimeInOut(source=sourcemixer, invalid_channels=[_ for _ in range(
        beamformer.steer.mics.num_mics) if not _ == ref_mic_idx]) # mask all channels except ref mic
    # get features
    data = {"loc" : array([s.loc for s in sourcemixer.sources], dtype=float32).T,
            "p2" : get_source_p2(sourcemixer, source_freq_data, fidx, None) }
    data.update(
        calc_input_features(input_features, freq_data, beamformer, fidx, f, num, cache_bf, cache_csm, cache_dir)
    )
    return data

def calc_input_features(input_features, freq_data, beamformer, fidx, f, num, cache_bf, cache_csm, cache_dir):
    data = {}
    if "csm" in input_features:
        data.update(
            {"csm": get_csm(freq_data=freq_data,
                            fidx=fidx,
                            cache_dir=cache_dir)
            })
    if "csmtriu" in input_features:
        data.update(
            {"csmtriu": get_nonredundant_csm(freq_data=freq_data,
                                            fidx=fidx,
                                            cache_dir=cache_dir)
            })
    if "sourcemap" in input_features:
        data.update(
            {"sourcemap": get_sourcemap(beamformer=beamformer,
                                            f=f,
                                            num=num,
                                            cache_dir=cache_dir)
            })
    if "eigmode" in input_features:
        data.update(
            {"eigmode": get_eigmode(freq_data=freq_data,
                                    fidx=fidx,
                                    cache_dir=cache_dir)
            })
    return data



if TF_FLAG:
    import tensorflow as tf

    from acoupipe.writer import WriteTFRecord, float_list_feature, int64_feature

    def save_tfrecord(self, split, size, name, startsample=1, tasks=1, progress_bar=True, address=None, cache_csm=False, cache_bf=False,
                    cache_dir=".", log=False):
        # setup process
        parallel = self._setup_generation_process(tasks, address, log, name)
        # get dataset pipeline that yields the data
        pipeline = self.build_pipeline(parallel, cache_csm, cache_bf, cache_dir)
        if parallel: pipeline.numworkers=tasks
        set_pipeline_seeds(pipeline, startsample, size, split)
        # create TFRecordWriter to save pipeline output to TFRecord File
        WriteTFRecord(name=name, source=pipeline,
                      encoder_funcs=self.get_encoder_funcs()).save(progress_bar)
    Dataset1.save_tfrecord = save_tfrecord


    def get_tf_dataset(self, split, size, startsample=1, tasks=1, progress_bar=False, address=None, cache_csm=False, cache_bf=False,
                        cache_dir=".", log=False):
        signature = {k: tf.TensorSpec(shape,dtype=tf.float32,name=k) for k, shape in self.get_feature_shapes().items()}
        return tf.data.Dataset.from_generator(
            partial(
                self.generate,
                split=split,size=size,startsample=startsample,tasks=tasks,progress_bar=progress_bar,
                cache_csm=cache_csm,cache_bf=cache_bf,cache_dir=cache_dir,address=address,log=log
                ) ,output_signature=signature)
    Dataset1.get_tf_dataset = get_tf_dataset

    def get_encoder_funcs(self):
        encoder_dict = {
            "idx": int64_feature,
            "seeds": float_list_feature,
        }
        encoder_dict.update({
            f : float_list_feature for f in self.get_dataset_feature_names() if f not in ["idx", "seeds"]
        })
        return encoder_dict
    Dataset1.get_encoder_funcs = get_encoder_funcs

