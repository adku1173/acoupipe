"""Contains classes for the generation of microphone array data from synthesized signals for acoustic testing applications.

    Currently, the following dataset generators are available:

    * :class:`DatasetSynthetic`: A simple and fast method that relies on synthetic white noise signals and spatially stationary sources radiating under anechoic conditions.

.. _default measurement setup:

.. figure:: ../../../../_static/msm_layout.png
    :width: 750
    :align: center

    Default measurement setup used in the :py:mod:`acoupipe.datasets.synthetic` module.

"""

from copy import deepcopy
from functools import partial

import acoular as ac
import numpy as np
from scipy.stats import norm, poisson
from traits.api import Bool, Dict, Either, Enum, Float, Instance, Int, List, observe

import acoupipe.sampler as sp
from acoupipe.config import TF_FLAG
from acoupipe.datasets.base import ConfigBase, DatasetBase
from acoupipe.datasets.features import (
    AnalyticNoiseStrengthFeature,
    AnalyticSourceStrengthFeature,
    BaseFeatureCollection,
    BaseFeatureCollectionBuilder,
    CSMFeature,
    CSMtriuFeature,
    EigmodeFeature,
    EstimatedNoiseStrengthFeature,
    EstimatedSourceStrengthFeature,
    LocFeature,
    SourcemapFeature,
    SpectrogramFeature,
    TargetmapFeature,
    TimeDataFeature,
)
from acoupipe.datasets.micgeom import tub_vogel64_ap1
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.utils import get_all_source_signals, get_uncorrelated_noise_source_recursively


class DatasetSynthetic(DatasetBase):
    r"""`DatasetSynthetic` is a purely synthetic microphone array source case generator.

    DatasetSynthetic relies on synthetic source signals from which the features are extracted and has been used in different publications,
    e.g. :cite:`Kujawski2019`, :cite:`Kujawski2022`, :cite:`Feng2022`. The default virtual simulation setup consideres a 64 channel microphone
    array and a planar observation area, as shown in the `default measurement setup`_ figure.

    **Default environmental properties**

    .. _Environmental Characteristics:

    .. table:: Default Environmental Characteristics

        ===================== ========================================
        Environment           Anechoic, Resting, Homogeneous Fluid
        Speed of sound        343 m/s
        Microphone Array      Vogel's spiral, :math:`M=64`, Aperture Size 1 m
        Observation Area      x,y in [-0.5,0.5], z=0.5
        Source Type           Monopole
        Source Signals        Uncorrelated White Noise (:math:`T=5\,s`)
        ===================== ========================================

    **Default FFT parameters**

    The underlying default FFT parameters are:

    .. table:: FFT Parameters

        ===================== ========================================
        Sampling Rate         He = 40, fs=13720 Hz
        Block size            128 Samples
        Block overlap         50 %
        Windowing             von Hann / Hanning
        ===================== ========================================


    **Default randomized properties**

    Several properties of the dataset are randomized for each source case when generating the data. Their respective distributions,
    are closely related to :cite:`Herold2017`. As such, the the microphone positions are spatially disturbed
    to account for uncertainties in the microphone placement. The number of sources, their positions, and strength is randomly chosen.
    Uncorrelated white noise is added to the microphone channels by default.

    .. table:: Randomized properties

        ==================================================================   ===================================================
        Sensor Position Deviation [m]                                        Bivariate normal distributed (:math:`\sigma = 0.001)`
        No. of Sources                                                       Poisson distributed (:math:`\lambda=3`)
        Source Positions [m]                                                 Bivariate normal distributed (:math:`\sigma = 0.1688`)
        Source Strength (:math:`[{Pa}^2]` at reference position)               Rayleigh distributed (:math:`\sigma_{R}=5`)
        Relative Noise Variance                                              Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
        ==================================================================   ===================================================

    Example
    -------

    .. code-block:: python

        from acoupipe.datasets.synthetic import DatasetSynthetic

        dataset = DatasetSynthetic()
        dataset_generator = dataset.generate_dataset(
            features=["sourcemap", "loc", "f", "num"], # choose the features to extract
            f=[1000,2000,3000], # choose the frequencies to extract
            split='training', # choose the split of the dataset
            size=10, # choose the size of the dataset
            )

        # get the first data sample
        data = next(dataset_generator)

        # print the keys of the dataset
        print(data.keys())


    **Initialization Parameters**
    """

    def __init__(self, mode="welch", mic_pos_noise=True, mic_sig_noise=True,
                snap_to_grid=False, random_signal_length=False, signal_length=5, fs=13720., min_nsources=1,
                max_nsources=10, tasks=1, logger=None, config=None):
        """Initialize the DatasetSynthetic object.

        The input parameters are passed to the DatasetSyntheticConfig object, which creates
        all necessary objects for the simulation of microphone array data.

        Parameters
        ----------
        mode : str
            Type of calculation method. Can be either :code:`welch`, :code:`analytic` or :code:`wishart`.
            Defaults to :code:`welch`.
        mic_pos_noise : bool
            Apply positional noise to microphone geometry. Defaults to True.
        mic_sig_noise : bool
            Apply additional uncorrelated white noise to microphone signals. Defaults to True.
        snap_to_grid : bool
            Snap source locations to grid. The grid is defined in the config object as
            config.grid. Defaults to False.
        random_signal_length : bool
            Randomize signal length. Defaults to False. If True, the signal length is
            uniformly sampled from the interval [1s,10s].
        signal_length : float
            Length of the signal in seconds. Defaults to 5 seconds.
        fs : float
            Sampling frequency in Hz. Defaults to 13720 Hz.
        min_nsources : int
            Minimum number of sources in the dataset. Defaults to 1.
        max_nsources : int
            Maximum number of sources in the dataset. Defaults to 10.
        tasks : int
            Number of parallel tasks. Defaults to 1.
        logger : logging.Logger
            Logger object. Defaults to None.
        config : DatasetSyntheticConfig
            Configuration object. Defaults to None. If None, a default configuration
            object is created.
        """
        if config is None:
            config = DatasetSyntheticConfig(
                mode=mode, signal_length=signal_length, fs=fs,
                min_nsources=min_nsources, max_nsources=max_nsources,
                mic_pos_noise=mic_pos_noise, mic_sig_noise=mic_sig_noise,
                snap_to_grid=snap_to_grid, random_signal_length=random_signal_length)
        super().__init__(config=config, tasks=tasks, logger=logger)

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        if f is None:
            fdim = self.config.freq_data.fftfreq().shape[0]
        elif isinstance(f, list):
            fdim = len(f)
        else:
            fdim = 1

        if self.config.random_signal_length:
            tdim = None
        else:
            tdim = int(self.config.signal_length*self.config.fs)

        builder = DatasetSyntheticFeatureCollectionBuilder(
            feature_collection = BaseFeatureCollection(),
            mdim = self.config.mics.num_mics,
            tdim = tdim,
            fdim = fdim,
        )
        # add prepare function
        builder.add_custom(self.config.get_prepare_func())
        builder.add_seeds(len(self.config.get_sampler()))
        builder.add_idx()
        # add feature functions
        if "time_data" in features:
            if self.config.mode == "welch":
                builder.add_time_data(self.config.freq_data.source)
            else:
                raise ValueError("time_data feature is not possible with modes ['analytic', 'wishart'].")
        if "spectrogram" in features:
            if self.config.mode == "welch":
                builder.add_spectrogram(self.config.fft_spectra, f, num)
            else:
                raise ValueError("spectrogram feature is not possible with modes ['analytic', 'wishart'].")
        if "csm" in features:
            builder.add_csm(self.config.freq_data, f, num)
        if "csmtriu" in features:
            builder.add_csmtriu(self.config.freq_data, f, num)
        if "eigmode" in features:
            builder.add_eigmode(self.config.freq_data, f, num)
        if "sourcemap" in features:
            builder.add_sourcemap(self.config.beamformer, f, num)
        if "loc" in features:
            builder.add_loc(self.config.freq_data)
        if "source_strength_analytic" in features:
            builder.add_source_strength_analytic(
                self.config.freq_data, f, num, steer=self.config.source_steer)
        if "source_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config.fft_obs_spectra
            else:
                freq_data = self.config.freq_data
            builder.add_source_strength_estimated(freq_data, f, num)
        if "noise_strength_analytic" in features:
            builder.add_noise_strength_analytic(self.config.freq_data, f, num)
        if "noise_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config.fft_spectra
            else:
                freq_data = self.config.freq_data
            builder.add_noise_strength_estimated(freq_data, f, num)
        if "targetmap_analytic" in features:
            builder.add_targetmap(self.config.freq_data, f, num, self.config.source_steer,
                ref_mic=None, strength_type="analytic", grid=self.config.grid)
        if "targetmap_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config.fft_obs_spectra
            else:
                freq_data = self.config.freq_data
            builder.add_targetmap(freq_data, f, num, self.config.source_steer,
                ref_mic=None, strength_type="estimated", grid=self.config.grid)
        if "f" in features:
            builder.add_f(self.config.freq_data.fftfreq(), f, num)
        if "num" in features:
            builder.add_num(num)
        return builder.build()


def sample_rms(nsources, rng):
    """Draw sources' squared rms pressures from Rayleigh distribution."""
    return np.sqrt(rng.rayleigh(5,nsources))

def sample_mic_noise_variance(rng):
    """Draw microphone noise variance, uniform distribution."""
    return rng.uniform(10e-6,0.1)

def sample_signal_seed(rng):
    return int(rng.uniform(1,1e9))

def sample_signal_length(rng):
    return rng.uniform(1,10)

class DatasetSyntheticConfig(ConfigBase):
    """
    Default Configuration class.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    signal_length : float
        Length of the source signals in seconds.
    max_nsources : int
        Maximum number of sources.
    min_nsources : int
        Minimum number of sources.
    mode : str
        Type of CSM calculation method.
    mic_pos_noise : bool
        Apply positional noise to microphone geometry.
    mic_sig_noise : bool
        Apply signal noise to microphone signals.
    snap_to_grid : bool
        Snap source locations to grid.
    random_signal_length : bool
        Randomize signal length (Default: uniformly sampled signal length [1s,10s]).
    fft_params : dict
        FFT parameters with default items :code:`block_size=128`,
        :code:`overlap="50%"`, :code:`window="Hanning"` and :code:`precision="complex64"`.
    env : ac.Environment
        Instance of acoular.Environment defining the environmental coditions,
        i.e. the speed of sound.
    mics : ac.MicGeom
        Instance of acoular.MicGeom defining the microphone array geometry.
    noisy_mics : ac.MicGeom
        a second instance of acoular.MicGeom defining the noisy microphone array geometry.
    obs : ac.MicGeom
        Instance of acoular.MicGeom defining the observation point which is used as the
        reference position when calculating the source strength.
    grid : ac.RectGrid
        Instance of acoular.RectGrid defining the grid on which the Beamformer calculates
        the source map and on which the targetmap feature is calculated.
    source_grid : ac.Grid
        Instance of acoular.Grid. Only relevant if :attr:`snap_to_grid` is :code:`True`.
        Then, the source locations are snapped to this grid. Default is a copy of :attr:`grid`.
    beamformer : ac.BeamformerBase
        Instance of acoular.BeamformerBase defining the beamformer used to calculate the sourcemap.
    steer : ac.SteeringVector
        Instance of acoular.SteeringVector defining the steering vector used to calculate the sourcemap.
    freq_data : ac.PowerSpectra
        Instance of acoular.PowerSpectra defining the frequency domain data. Only used if :attr:`mode` is
        :code:`welch`. Otherwise, an instance of :class:`acoupipe.datasets.spectra_analytic.PowerSpectraAnalytic`
        is used.
    fft_spectra : ac.RFFT
        Instance of acoular.RFFT used to calculate the spectrogram data. Only used if :attr:`mode` is
        :code:`welch`.
    fft_obs_spectra : ac.PowerSpectra
        Instance of acoular.PowerSpectra used to calculate the source strength at the observation point given in
        :attr:`obs`.
    signals : list
        List of signals.
    sources : list
        List of sources.
    mic_noise_signal : ac.SignalGenerator
        Noise signal configuration object.
    mic_noise_source : ac.UncorrelatedNoiseSource
        Noise source configuration object.
    micgeom_sampler : sp.MicGeomSampler
        Sampler that applies positional noise to the microphone geometry.
    location_sampler : sp.LocationSampler
        Source location sampler that samples the locations of the sound sources.
    rms_sampler : sp.ContainerSampler
        Signal RMS sampler that samples the RMS values of the source signals.
    nsources_sampler : sp.NumericAttributeSampler
        Number of sources sampler.
    mic_noise_sampler : sp.ContainerSampler
        Microphone noise sampler that creates random uncorrelated noise at the microphones.
    signal_length_sampler : sp.ContainerSampler
        Signal length sampler that samples the length of the source signals. Only used if :attr:`random_signal_length` is :code:`True`.
    """

    # public traits
    fs = Float(13720, desc="sampling frequency")
    signal_length = Float(5, desc="length of the signal in seconds")
    max_nsources = Int(10, desc="maximum number of sources")
    min_nsources = Int(1, desc="minimum number of sources")
    mode = Enum(("welch", "analytic", "wishart"), default="welch",
                            desc="type of PowerSpectra calculation method.")
    mic_pos_noise = Bool(True, desc="apply positional noise to microphone geometry")
    mic_sig_noise = Bool(True, desc="apply signal noise to microphone signals")
    snap_to_grid = Bool(False, desc="snap source locations to grid")
    random_signal_length = Bool(False, desc="randomize signal length")
    fft_params = Dict({
                    "block_size" : 128,
                    "overlap" : "50%",
                    "window" : "Hanning",
                    "precision" : "complex64"},
                desc="FFT parameters")
    # acoular pipeline traits
    env = Instance(ac.Environment, desc="environment configuration")
    mics = Instance(ac.MicGeom, desc="microphone geometry configuration")
    noisy_mics = Instance(ac.MicGeom, desc="microphone geometry configuration")
    obs = Instance(ac.MicGeom, desc="observation point configuration")
    grid = Instance(ac.RectGrid, desc="grid configuration")
    source_grid = Instance(ac.Grid, desc="source grid configuration (only relevant if snap_to_grid=True)")
    beamformer = Instance(ac.BeamformerBase, desc="beamformer configuration")
    steer = Instance(ac.SteeringVector, desc="steering vector configuration")
    freq_data = Instance(ac.PowerSpectra, desc="frequency domain data configuration")
    fft_spectra = Instance(ac.RFFT, desc="FFT spectra configuration (only for spectrogram feature)")
    fft_obs_spectra = Instance(ac.PowerSpectra, desc="FFT spectra configuration only for 'estimated strength' label)")
    signals = List(desc="list of signals")
    sources = List(desc="list of sound sources")
    mic_noise_signal = Instance(ac.SignalGenerator, desc="noise signal configuration")
    mic_noise_source = Instance(ac.UncorrelatedNoiseSource, desc="noise source configuration")
    source_steer = Instance(ac.SteeringVector, desc="steering vector configuration")

    # sampler traits
    micgeom_sampler = Instance(sp.MicGeomSampler,
            desc="microphone geometry positional noise sampler")
    location_sampler = Instance(sp.LocationSampler, desc="source location sampler")
    signal_seed_sampler = Instance(sp.ContainerSampler, desc="signal seed sampler")
    rms_sampler = Instance(sp.ContainerSampler, desc="signal rms sampler")
    nsources_sampler = Instance(sp.NumericAttributeSampler,
                    desc="number of sources sampler")
    mic_noise_sampler = Instance(sp.ContainerSampler,
                    desc="microphone noise sampler")
    signal_length_sampler = Instance(sp.ContainerSampler,
                    desc="signal length sampler (only if random_signal_length=True)")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_acoular_pipeline()

    @observe("mode, signal_length, fs, max_nsources, fft_params.items, mic_sig_noise")
    def recreate_acoular_pipeline(self, event):
        self.create_acoular_pipeline()

    def create_acoular_pipeline(self):
        self.env = self.create_env()
        self.mics = self.create_mics()
        self.noisy_mics = self.create_mics()
        self.grid = self.create_grid()
        self.source_grid = self.create_source_grid()
        self.steer = self.create_steer()
        self.obs = self.create_obs()
        self.signals = self.create_signals()
        self.sources = self.create_sources()
        self.source_steer = self.create_source_steer()
        self.mic_noise_signal = self.create_mic_noise_signal()
        self.mic_noise_source = self.create_mic_noise_source()
        self.freq_data = self.create_freq_data()
        self.fft_spectra = self.create_fft_spectra()
        self.fft_obs_spectra = self.create_fft_obs_spectra()
        self.beamformer = self.create_beamformer()

    def create_sampler(self):
        self.micgeom_sampler = self.create_micgeom_sampler()
        self.location_sampler = self.create_location_sampler()
        self.signal_seed_sampler = self.create_signal_seed_sampler()
        self.rms_sampler = self.create_rms_sampler()
        self.nsources_sampler = self.create_nsources_sampler()
        self.mic_noise_sampler = self.create_mic_noise_sampler()
        self.signal_length_sampler = self.create_signal_length_sampler()

    def get_sampler(self):
        self.create_sampler()
        sampler = {
            2 : self.signal_seed_sampler,
            3 : self.rms_sampler,
            4 : self.location_sampler,
            }

        if self.max_nsources != self.min_nsources:
            sampler[0] = self.nsources_sampler
        if self.mic_pos_noise:
            sampler[1] = self.micgeom_sampler
        if self.mic_sig_noise:
            sampler[5] = self.mic_noise_sampler
        if self.random_signal_length:
            sampler[6] = self.signal_length_sampler
        return sampler

    def create_env(self):
        return ac.Environment(c=343.0)

    def create_mics(self):
        return ac.MicGeom(pos_total = tub_vogel64_ap1)

    def create_grid(self):
        ap = self.mics.aperture
        return ac.RectGrid(y_min=-0.5*ap, y_max=0.5*ap, x_min=-0.5*ap, x_max=0.5*ap,
                                    z=0.5*ap, increment=1/63*ap)

    def create_source_grid(self):
        return self.create_grid()

    def create_steer(self):
        return ac.SteeringVector(
            steer_type="true level",
            ref=tub_vogel64_ap1[:,63], # centermost mic,
            mics=self.mics,
            grid=self.grid,
            env=self.env)

    def create_obs(self):
        return ac.MicGeom(
            pos_total=self.steer.ref[:,np.newaxis])

    def create_beamformer(self):
        return ac.BeamformerBase(
            r_diag=False,
            precision="float32",
            cached=False,
            freq_data = self.freq_data,
            steer = self.steer,
            )

    def create_signals(self):
        signals = []
        for i in range(self.max_nsources):
            signals.append(ac.WNoiseGenerator(
                    seed = i+1,
                    sample_freq=self.fs,
                    num_samples=self.signal_length*self.fs,
                    )
            )
        return signals

    def create_sources(self):
        sources = []
        for signal in self.signals:
            sources.append(
                ac.PointSource(
                    signal=signal,
                    mics=self.noisy_mics,
                    env=self.env,
                    )
            )
        return sources

    def create_fft_spectra(self):
        if self.mic_sig_noise:
            source = ac.Mixer(source=self.mic_noise_source,
                            sources=self.sources,)
        else:
            source = ac.SourceMixer(sources=self.sources)
        return ac.RFFT(
            source = source,
            **self.fft_params,
            )

    def create_fft_obs_spectra(self):
        return ac.PowerSpectra(
            source = ac.SourceMixer(sources=self.sources),
            cached = False,
            **self.fft_params,
            )

    def create_freq_data(self):
        if self.mode == "welch":
            if self.mic_sig_noise:
                source = ac.Mixer(source=self.mic_noise_source,
                                sources=self.sources,)
            else:
                source = ac.SourceMixer(sources=self.sources)
            return ac.PowerSpectra(
                    cached = False,
                    source = source,
                    **self.fft_params,
                    )
        else:
            fft_params = deepcopy(self.fft_params)
            fft_params.pop("window")
            return PowerSpectraAnalytic(
                mode = self.mode,
                num_samples=self.signal_length*self.fs,
                sample_freq=self.fs,
                steer=self.source_steer,
                cached = False,
                **fft_params
                )

    def create_mic_noise_signal(self):
        return ac.WNoiseGenerator(
                seed = 1000,
                sample_freq=self.fs,
                num_samples=self.signal_length*self.fs,
                )

    def create_mic_noise_source(self):
        return ac.UncorrelatedNoiseSource(
                signal=self.mic_noise_signal,
                mics=self.noisy_mics,
                )

    def create_source_steer(self):
        return ac.SteeringVector(
            steer_type="true level",
            ref=self.obs.pos.squeeze(),
            mics=self.noisy_mics,
            grid=ac.ImportGrid(), # is filled later
            env=self.env
            )

    def create_micgeom_sampler(self):
        return sp.MicGeomSampler(
            random_var = norm(loc=0, scale=0.001),
            ddir = np.array([[1.0], [1.0], [0]]),
            target = self.noisy_mics,
            mpos_init = self.mics.pos_total,)

    def create_location_sampler(self):
        ap = self.mics.aperture
        z = self.grid.z
        location_sampler = sp.LocationSampler(
            random_var = (norm(0,0.1688*ap),norm(0,0.1688*ap),norm(z,0)),
            x_bounds = (-0.5*ap,0.5*ap),
            y_bounds = (-0.5*ap,0.5*ap),
            z_bounds = (0.5*ap,0.5*ap),
            nsources = self.max_nsources,)
        if self.snap_to_grid:
            location_sampler.grid = self.source_grid
        return location_sampler

    def create_rms_sampler(self):
        random_func = partial(sample_rms, self.max_nsources)
        return sp.ContainerSampler(
            random_func = random_func)

    def create_signal_seed_sampler(self):
        return sp.ContainerSampler(
            random_func = sample_signal_seed)

    def create_nsources_sampler(self):
        return sp.NumericAttributeSampler(
            random_var = poisson(mu=3, loc=1),
            attribute = "nsources",
            equal_value = True,
            target=[self.location_sampler],
            filter=lambda x: (x <= self.max_nsources) and (
                x >= self.min_nsources))

    def create_mic_noise_sampler(self):
        return sp.ContainerSampler(
            random_func = sample_mic_noise_variance)

    def create_signal_length_sampler(self):
        return sp.ContainerSampler(
            random_func = sample_signal_length)

    @staticmethod
    def calc_welch_prepare_func(sampler, beamformer, sources, source_steer, fft_spectra, fft_obs_spectra, obs):
        # restore sampler and acoular objects
        micgeom_sampler = sampler.get(1)
        seed_sampler = sampler.get(2)
        rms_sampler = sampler.get(3)
        loc_sampler = sampler.get(4)
        noise_sampler = sampler.get(5)
        signal_length_sampler = sampler.get(6)

        freq_data = beamformer.freq_data

        if micgeom_sampler is not None:
            noisy_mics = micgeom_sampler.target
        else:
            noisy_mics = beamformer.steer.mics # use the original mics (without noise)

        if signal_length_sampler is not None:
            # adjust source signals, noise signal length
            signals = get_all_source_signals(sources)
            for signal in signals:
                signal.num_samples = signal_length_sampler.target*freq_data.sample_freq
        # sample parameters
        loc = loc_sampler.target
        nsources = loc.shape[1]
        prms_sq = rms_sampler.target[:nsources]**2 # squared sound pressure RMS at reference position
        # apply parameters
        mic_noise = get_uncorrelated_noise_source_recursively(freq_data.source)
        if mic_noise:
            mic_noise_signal = mic_noise[0].signal
            if signal_length_sampler is not None:
                mic_noise_signal.num_samples = signal_length_sampler.target*freq_data.sample_freq
            if noise_sampler is not None:
                noise_signal_ratio = noise_sampler.target # normalized noise variance
                noise_prms_sq = prms_sq.sum()*noise_signal_ratio
                mic_noise_signal.rms = np.sqrt(noise_prms_sq)
                mic_noise_signal.seed = seed_sampler.target+1000
                freq_data.source.source.mics = noisy_mics
        subset_sources = sources[:nsources]
        source_steer.grid = ac.ImportGrid(pos=loc) # set source locations
        for i,src in enumerate(subset_sources):
            src.signal.seed = seed_sampler.target+i
            # weight the RMS with the distance to the reference position
            src.signal.rms = np.sqrt(prms_sq[i])*source_steer.r0[i]
            src.loc = (loc[0,i], loc[1,i], loc[2,i]) # apply wishart locations
            src.mics = noisy_mics
        freq_data.source.sources = subset_sources # apply subset of sources
        fft_spectra.source = freq_data.source # only for spectrogram feature
        # update observation point
        obs_sources = deepcopy(subset_sources)
        for src in obs_sources:
            src.mics = obs
        fft_obs_spectra.source = ac.SourceMixer(sources=obs_sources)
        return {}

    @staticmethod
    def calc_analytic_prepare_func(sampler, beamformer):
        # restore sampler and acoular objects
        micgeom_sampler = sampler.get(1)
        seed_sampler = sampler.get(2)
        rms_sampler = sampler.get(3)
        loc_sampler = sampler.get(4)
        noise_sampler = sampler.get(5)
        signal_length_sampler = sampler.get(6)

        freq_data = beamformer.freq_data

        if micgeom_sampler is not None:
            noisy_mics = micgeom_sampler.target
        else:
            noisy_mics = beamformer.steer.mics # use the original mics (without noise)

        if signal_length_sampler is not None:
            freq_data.num_samples = signal_length_sampler.target*freq_data.sample_freq

        nfft = freq_data.fftfreq().shape[0]
        # sample parameters
        loc = loc_sampler.target
        nsources = loc.shape[1]
        freq_data.steer.grid = ac.ImportGrid(pos=loc) # set source locations
        freq_data.steer.mics = noisy_mics # set mic locations
        freq_data.seed=seed_sampler.target
        # change source strength
        prms_sq = rms_sampler.target[:nsources]**2 # squared sound pressure RMS at reference position
        prms_sq_per_freq = prms_sq / nfft #prms_sq_per_freq
        freq_data.Q = np.stack([np.diag(prms_sq_per_freq) for _ in range(nfft)], axis=0)
        # add noise to freq_data
        if noise_sampler is not None:
            noise_signal_ratio = noise_sampler.target # normalized noise variance
            noise_prms_sq = prms_sq.sum()*noise_signal_ratio
            noise_prms_sq_per_freq = noise_prms_sq / nfft
            nperf = np.diag(np.array([noise_prms_sq_per_freq]*beamformer.steer.mics.num_mics))
            freq_data.noise = np.stack([nperf for _ in range(nfft)], axis=0)
        else:
            freq_data.noise = None
        return {}

    def get_prepare_func(self):
        if self.mode == "welch":
            prepare_func = partial(
            self.calc_welch_prepare_func,
            beamformer=self.beamformer,
            sources=self.sources,
            source_steer = self.source_steer,
            fft_spectra=self.fft_spectra,
            fft_obs_spectra=self.fft_obs_spectra,
            obs = self.obs)
        else:
            prepare_func = partial(
            self.calc_analytic_prepare_func,
            beamformer=self.beamformer)
        return prepare_func


class DatasetSyntheticFeatureCollectionBuilder(BaseFeatureCollectionBuilder):

    tdim = Either(Int(desc="time dimension"), None)
    fdim = Int(desc="frequency dimension")
    mdim = Int(desc="microphone dimension")

    def add_time_data(self, time_data):
        """
        Add a time_data feature to the BaseFeatureCollection.

        Parameters
        ----------
        time_data : str
            source object containing the time data e.g. ac.TimeSamples class instance.
        """
        calc_time_data = TimeDataFeature(
            time_data=time_data, dtype=np.float32).get_feature_func()
        self.feature_collection.add_feature_func(calc_time_data)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"time_data" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"time_data" : (self.tdim,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"time_data" : "float32"})

    def add_spectrogram(self, freq_data, f, num):
        calc_spectrogram = SpectrogramFeature(freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_spectrogram)
        if TF_FLAG:
            from acoupipe.writer import complex_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"spectrogram" : complex_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"spectrogram" : (None,self.fdim,self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"spectrogram" : "complex64"})

    def add_csm(self, freq_data, f, num):
        calc_csm = CSMFeature(freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_csm)
        if TF_FLAG:
            from acoupipe.writer import complex_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"csm" : complex_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"csm" : (self.fdim,self.mdim,self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"csm" : "complex64"})

    def add_csmtriu(self, freq_data, f, num):
        calc_csmtriu = CSMtriuFeature(freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_csmtriu)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"csmtriu" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"csmtriu" : (self.fdim,self.mdim,self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"csmtriu" : "float32"})

    def add_eigmode(self, freq_data, f, num):
        calc_eigmode = EigmodeFeature(freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_eigmode)
        if TF_FLAG:
            from acoupipe.writer import complex_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"eigmode" : complex_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"eigmode" : (self.fdim,self.mdim,self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"eigmode" : "complex64"})

    def add_sourcemap(self, beamformer, f, num):
        calc_sourcemap = SourcemapFeature(beamformer=beamformer, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_sourcemap)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"sourcemap" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"sourcemap" : (self.fdim,) + beamformer.steer.grid.shape})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"sourcemap" : "float32"})

    def add_loc(self, freq_data):
        calc_loc = LocFeature(freq_data=freq_data).get_feature_func()
        self.feature_collection.add_feature_func(calc_loc)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"loc" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"loc" : (3,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"loc" : "float32"})

    def add_source_strength_analytic(self, freq_data, f, num, steer):
        calc_strength = AnalyticSourceStrengthFeature(
            freq_data=freq_data, f=f, num=num, steer=steer).get_feature_func()
        self.feature_collection.add_feature_func(calc_strength)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"source_strength_analytic" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"source_strength_analytic" : (self.fdim,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"source_strength_analytic" : "float32"})

    def add_source_strength_estimated(self, freq_data, f, num):
        calc_strength = EstimatedSourceStrengthFeature(
            freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_strength)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"source_strength_estimated" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"source_strength_estimated" : (self.fdim,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"source_strength_estimated" : "float32"})

    def add_noise_strength_analytic(self, freq_data, f, num):
        calc_noise = AnalyticNoiseStrengthFeature(
            freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_noise)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"noise_strength_analytic" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"noise_strength_analytic" : (self.fdim, self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"noise_strength_analytic" : "float32"})

    def add_noise_strength_estimated(self, freq_data, f, num):
        calc_noise = EstimatedNoiseStrengthFeature(
            freq_data=freq_data, f=f, num=num).get_feature_func()
        self.feature_collection.add_feature_func(calc_noise)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"noise_strength_estimated" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"noise_strength_estimated" : (self.fdim, self.mdim)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"noise_strength_estimated" : "float32"})

    def add_targetmap(self, freq_data, f, num, steer, ref_mic, strength_type, grid):
        name = f"targetmap_{strength_type}"
        calc_targetmap = TargetmapFeature(
            freq_data=freq_data, f=f, num=num, steer=steer, ref_mic=ref_mic,
            strength_type=strength_type, grid=grid, name=name).get_feature_func()
        self.feature_collection.add_feature_func(calc_targetmap)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {name : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {name : (self.fdim,) + grid.shape})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {name : "float32"})

    def add_seeds(self, nsampler):
        if TF_FLAG:
            from acoupipe.writer import int_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "seeds" : int_list_feature,})
            self.feature_collection.feature_tf_shape_mapper.update({
                                            "seeds" : (nsampler,2)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"seeds" : "int64"})

    def add_idx(self):
        if TF_FLAG:
            from acoupipe.writer import int64_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "idx" : int64_feature,})
            self.feature_collection.feature_tf_shape_mapper.update({
                                            "idx" : (),})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"idx" : "int64"})

    def add_f(self, fftfreq, f, num):
        if f is None:
            all_f = fftfreq
        elif isinstance(f, list):
            if num == 0:
                all_f = np.array(
                    [fftfreq[np.searchsorted(fftfreq, freq)] for freq in f])
            else:
                all_f = np.array(f)
        else:
            all_f = np.array([fftfreq[np.searchsorted(fftfreq, f)]])

        def get_f(sampler, f):
            return {"f": f}
        feature_func = partial(get_f, f=all_f)
        self.feature_collection.add_feature_func(feature_func)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "f" : float_list_feature,})
            self.feature_collection.feature_tf_shape_mapper.update({
                                            "f" : (self.fdim,),})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"f" : "float32"})
    def add_num(self, num):
        def add_num(sampler, num):
            return {"num": num}
        self.feature_collection.add_feature_func(partial(add_num, num=num))
        if TF_FLAG:
            from acoupipe.writer import int64_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "num" : int64_feature,})
            self.feature_collection.feature_tf_shape_mapper.update({
                                            "num" : (),})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"num" : "int64"})


class DatasetSyntheticTestConfig(DatasetSyntheticConfig):

    def create_mics(self):
        return ac.MicGeom(pos_total=np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
        [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
        [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]]))

    def create_grid(self):
        ap = self.mics.aperture
        return ac.RectGrid(y_min=-0.5*ap, y_max=0.5*ap, x_min=-0.5*ap, x_max=0.5*ap,
                                    z=0.5*ap, increment=1/5*ap)










