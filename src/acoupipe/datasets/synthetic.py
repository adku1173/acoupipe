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
from traits.api import Bool, Dict, Enum, Float, Instance, Int, List, observe

import acoupipe.sampler as sp
from acoupipe.datasets.base import ConfigBase, DatasetBase
from acoupipe.datasets.features import (
    AnalyticNoiseStrengthFeature,
    AnalyticSourceStrengthFeature,
    BaseFeatureCatalog,
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
    create_feature,
)
from acoupipe.datasets.ir import get_ir, require_ir_support
from acoupipe.datasets.micgeom import tub_vogel64_ap1
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.utils import calc_transfer, get_all_source_signals, get_uncorrelated_noise_source_recursively


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
            features=['sourcemap', 'loc', 'f', 'num'],  # choose the features to extract
            f=[1000, 2000, 3000],  # choose the frequencies to extract
            split='training',  # choose the split of the dataset
            size=10,  # choose the size of the dataset
        )

        # get the first data sample
        data = next(dataset_generator)

        # print the keys of the dataset
        print(data.keys())


    **Initialization Parameters**
    """

    def __init__(
        self,
        mode='welch',
        mic_pos_noise=True,
        mic_sig_noise=True,
        snap_to_grid=False,
        random_signal_length=False,
        signal_length=5,
        fs=13720.0,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        remote_args=None,
        logger=None,
        config=None,
    ):
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
        remote_args : dict
            Dictionary of keyword arguments passed to the remote actors when using Ray for parallelization. Defaults to None.
        logger : logging.Logger
            Logger object. Defaults to None.
        config : DatasetSyntheticConfig
            Configuration object. Defaults to None. If None, a default configuration
            object is created.
        """
        if config is None:
            config = DatasetSyntheticConfig(
                mode=mode,
                signal_length=signal_length,
                fs=fs,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                mic_pos_noise=mic_pos_noise,
                mic_sig_noise=mic_sig_noise,
                snap_to_grid=snap_to_grid,
                random_signal_length=random_signal_length,
            )
        super().__init__(config=config, tasks=tasks, logger=logger, remote_args=remote_args)

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        # handle all custom features (BaseFeatureCatalog instances)
        custom_features = [feat for feat in features if isinstance(feat, BaseFeatureCatalog)]
        # collect default features defined by name
        default_feature_names = [feat for feat in features if isinstance(feat, str)]
        default_features = self.config.get_default_features(default_feature_names, f, num)
        builder = BaseFeatureCollectionBuilder(features=default_features + custom_features)
        builder.add_custom(self.config.get_prepare_func())  # add prepare function
        feature_collection = builder.build()  # finally build the feature collection
        builder.add_custom(self.config.get_cleanup_func(features))  # add cleanup function
        return feature_collection


def sample_rms(nsources, rng):
    """Draw sources' squared rms pressures from Rayleigh distribution."""
    return np.sqrt(rng.rayleigh(5, nsources))


def sample_mic_noise_variance(rng):
    """Draw microphone noise variance, uniform distribution."""
    return rng.uniform(10e-6, 0.1)


def sample_signal_seed(rng):
    return int(rng.uniform(1, 1e9))


def sample_signal_length(rng):
    return rng.uniform(1, 10)


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
    fs = Float(13720, desc='sampling frequency')
    signal_length = Float(5, desc='length of the signal in seconds')
    max_nsources = Int(10, desc='maximum number of sources')
    min_nsources = Int(1, desc='minimum number of sources')
    mode = Enum(('welch', 'analytic', 'wishart'), default='welch', desc='type of PowerSpectra calculation method.')
    mic_pos_noise = Bool(True, desc='apply positional noise to microphone geometry')
    mic_sig_noise = Bool(True, desc='apply signal noise to microphone signals')
    snap_to_grid = Bool(False, desc='snap source locations to grid')
    random_signal_length = Bool(False, desc='randomize signal length')
    fft_params = Dict(
        {'block_size': 128, 'overlap': '50%', 'window': 'Hanning', 'precision': 'complex64'},
        desc='FFT parameters',
    )
    # acoular pipeline traits
    env = Instance(ac.Environment, desc='environment configuration')
    mics = Instance(ac.MicGeom, desc='microphone geometry configuration')
    noisy_mics = Instance(ac.MicGeom, desc='microphone geometry configuration')
    obs = Instance(ac.MicGeom, desc='observation point configuration')
    grid = Instance(ac.RectGrid, desc='grid configuration')
    source_grid = Instance(ac.Grid, desc='source grid configuration (only relevant if snap_to_grid=True)')
    beamformer = Instance(ac.BeamformerBase, desc='beamformer configuration')
    steer = Instance(ac.SteeringVector, desc='steering vector configuration')
    freq_data = Instance(ac.PowerSpectra, desc='frequency domain data configuration')
    fft_spectra = Instance(ac.RFFT, desc='FFT spectra configuration (only for spectrogram feature)')
    fft_obs_spectra = Instance(ac.PowerSpectra, desc="FFT spectra configuration only for 'estimated strength' label)")
    signals = List(desc='list of signals')
    sources = List(desc='list of sound sources')
    mic_noise_signal = Instance(ac.SignalGenerator, desc='noise signal configuration')
    mic_noise_source = Instance(ac.UncorrelatedNoiseSource, desc='noise source configuration')
    source_steer = Instance(ac.SteeringVector, desc='steering vector configuration')

    # sampler traits
    micgeom_sampler = Instance(sp.MicGeomSampler, desc='microphone geometry positional noise sampler')
    location_sampler = Instance(sp.LocationSampler, desc='source location sampler')
    signal_seed_sampler = Instance(sp.ContainerSampler, desc='signal seed sampler')
    rms_sampler = Instance(sp.ContainerSampler, desc='signal rms sampler')
    nsources_sampler = Instance(sp.NumericAttributeSampler, desc='number of sources sampler')
    mic_noise_sampler = Instance(sp.ContainerSampler, desc='microphone noise sampler')
    signal_length_sampler = Instance(
        sp.ContainerSampler,
        desc='signal length sampler (only if random_signal_length=True)',
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_acoular_pipeline()

    @observe('mode, signal_length, fs, max_nsources, fft_params.items, mic_sig_noise')
    def recreate_acoular_pipeline(self, event):  # noqa ARG002
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
            2: self.signal_seed_sampler,
            3: self.rms_sampler,
            4: self.location_sampler,
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

    def _get_fftfreq(self):
        return self.freq_data.fftfreq()

    def _get_fdim(self, f):
        fftfreq = self._get_fftfreq()
        if f is None:
            return fftfreq.shape[0]
        if isinstance(f, list):
            return len(f)
        return 1

    def _get_mdim(self):
        return self.mics.num_mics

    def _get_tdim(self):
        return None if self.random_signal_length else int(self.signal_length * self.fs)

    def _get_default_feature_kwargs(self, f, num):
        """Return keyword arguments passed to default feature builder methods."""
        return {
            'f': f,
            'num': num,
            'fdim': self._get_fdim(f),
            'mdim': self._get_mdim(),
            'tdim': self._get_tdim(),
            'fftfreq': self._get_fftfreq(),
        }

    def _get_default_feature_time_data(self, **kwargs):  # noqa ARG002
        if self.mode != 'welch':
            msg = "time_data feature is not possible with modes ['analytic', 'wishart']."
            raise ValueError(msg)
        return TimeDataFeature(time_data=self.freq_data.source, dtype=np.float32, shape=(kwargs['tdim'], None))

    def _get_default_feature_spectrogram(self, **kwargs):  # noqa ARG002
        if self.mode != 'welch':
            msg = "spectrogram feature is not possible with modes ['analytic', 'wishart']."
            raise ValueError(msg)
        return SpectrogramFeature(
            freq_data=self.fft_spectra,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.complex64,
            shape=(None, kwargs['fdim'], kwargs['mdim']),
        )

    def _get_default_feature_csm(self, **kwargs):  # noqa ARG002
        return CSMFeature(
            freq_data=self.freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.complex64,
            shape=(kwargs['fdim'], kwargs['mdim'], kwargs['mdim']),
        )

    def _get_default_feature_csmtriu(self, **kwargs):  # noqa ARG002
        return CSMtriuFeature(
            freq_data=self.freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'], kwargs['mdim'], kwargs['mdim']),
        )

    def _get_default_feature_eigmode(self, **kwargs):  # noqa ARG002
        return EigmodeFeature(
            freq_data=self.freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.complex64,
            shape=(kwargs['fdim'], kwargs['mdim'], kwargs['mdim']),
        )

    def _get_default_feature_sourcemap(self, **kwargs):  # noqa ARG002
        return SourcemapFeature(
            beamformer=self.beamformer,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'],) + self.beamformer.steer.grid.shape,
        )

    def _get_default_feature_loc(self, **kwargs):  # noqa ARG002
        return LocFeature(dtype=np.float32, shape=(3, None))

    def _get_default_feature_source_strength_analytic(self, **kwargs):  # noqa ARG002
        return AnalyticSourceStrengthFeature(
            freq_data=self.freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'], None),
        )

    def _get_default_feature_source_strength_estimated(self, **kwargs):  # noqa ARG002
        freq_data = self.fft_obs_spectra if self.mode == 'welch' else self.freq_data
        return EstimatedSourceStrengthFeature(
            freq_data=freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'], None),
        )

    def _get_default_feature_noise_strength_analytic(self, **kwargs):  # noqa ARG002
        return AnalyticNoiseStrengthFeature(
            freq_data=self.freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'], kwargs['mdim']),
        )

    def _get_default_feature_noise_strength_estimated(self, **kwargs):  # noqa ARG002
        freq_data = self.fft_spectra if self.mode == 'welch' else self.freq_data
        return EstimatedNoiseStrengthFeature(
            freq_data=freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            dtype=np.float32,
            shape=(kwargs['fdim'], kwargs['mdim']),
        )

    def _get_targetmap_feature(self, strength_type, **kwargs):  # noqa ARG002
        freq_data = (
            self.freq_data
            if strength_type == 'analytic'
            else (self.fft_obs_spectra if self.mode == 'welch' else self.freq_data)
        )
        return TargetmapFeature(
            freq_data=freq_data,
            f=kwargs['f'],
            num=kwargs['num'],
            ref_mic=None,
            strength_type=strength_type,
            grid=self.grid,
            name=f'targetmap_{strength_type}',
            dtype=np.float32,
            shape=(kwargs['fdim'],) + self.grid.shape,
        )

    def _get_default_feature_targetmap_analytic(self, **kwargs):  # noqa ARG002
        return self._get_targetmap_feature('analytic', **kwargs)

    def _get_default_feature_targetmap_estimated(self, **kwargs):  # noqa ARG002
        return self._get_targetmap_feature('estimated', **kwargs)

    def _get_default_feature_f(self, **kwargs):  # noqa ARG002
        if kwargs['f'] is None:
            all_f = kwargs['fftfreq']
        elif isinstance(kwargs['f'], list):
            if kwargs['num'] == 0:
                all_f = np.array([kwargs['fftfreq'][np.searchsorted(kwargs['fftfreq'], freq)] for freq in kwargs['f']])
            else:
                all_f = np.array(kwargs['f'])
        else:
            all_f = np.array([kwargs['fftfreq'][np.searchsorted(kwargs['fftfreq'], kwargs['f'])]])

        def get_f(sampler, f):  # noqa ARG001
            return {'f': f}

        return create_feature(
            feature_func=partial(get_f, f=all_f),
            name='f',
            shape=(kwargs['fdim'],),
            dtype=np.float32,
        )

    def _get_default_feature_num(self, **kwargs):  # noqa ARG002
        def add_num(sampler, num):  # noqa ARG001
            return {'num': num}

        return create_feature(
            feature_func=partial(add_num, num=kwargs['num']),
            name='num',
            shape=(),
            dtype=np.int64,
        )

    def create_env(self):
        return ac.Environment(c=343.0)

    def create_mics(self):
        return ac.MicGeom(pos_total=tub_vogel64_ap1)

    def create_grid(self):
        # round of numerical errors
        ap = np.round(self.mics.aperture, decimals=12)
        return ac.RectGrid(
            y_min=-0.5 * ap,
            y_max=0.5 * ap,
            x_min=-0.5 * ap,
            x_max=0.5 * ap,
            z=0.5 * ap,
            increment=1 / 63 * ap,
        )

    def create_source_grid(self):
        return self.create_grid()

    def create_steer(self):
        return ac.SteeringVector(
            steer_type='true level',
            ref=tub_vogel64_ap1[:, 63],  # centermost mic,
            mics=self.mics,
            grid=self.grid,
            env=self.env,
        )

    def create_obs(self):
        return ac.MicGeom(pos_total=self.steer.ref[:, np.newaxis])

    def create_beamformer(self):
        return ac.BeamformerBase(
            r_diag=False,
            precision='float32',
            cached=False,
            freq_data=self.freq_data,
            steer=self.steer,
        )

    def create_signals(self):
        signals = []
        for i in range(self.max_nsources):
            signals.append(
                ac.WNoiseGenerator(
                    seed=i + 1,
                    sample_freq=self.fs,
                    num_samples=self.signal_length * self.fs,
                ),
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
                ),
            )
        return sources

    def create_fft_spectra(self):
        if self.mic_sig_noise:
            source = ac.Mixer(source=self.mic_noise_source, sources=self.sources)
        else:
            source = ac.SourceMixer(sources=self.sources)
        return ac.RFFT(
            source=source,
            **self.fft_params,
        )

    def create_fft_obs_spectra(self):
        return ac.PowerSpectra(
            source=ac.SourceMixer(sources=self.sources),
            cached=False,
            **self.fft_params,
        )

    def create_freq_data(self):
        if self.mode == 'welch':
            if self.mic_sig_noise:
                source = ac.Mixer(source=self.mic_noise_source, sources=self.sources)
            else:
                source = ac.SourceMixer(sources=self.sources)
            return ac.PowerSpectra(
                cached=False,
                source=source,
                **self.fft_params,
            )
        return PowerSpectraAnalytic(
            mode=self.mode,
            num_samples=self.signal_length * self.fs,
            sample_freq=self.fs,
            steer=self.source_steer,
            cached=False,
            **self.fft_params,
        )

    def create_mic_noise_signal(self):
        return ac.WNoiseGenerator(
            seed=1000,
            sample_freq=self.fs,
            num_samples=self.signal_length * self.fs,
        )

    def create_mic_noise_source(self):
        return ac.UncorrelatedNoiseSource(
            signal=self.mic_noise_signal,
            mics=self.noisy_mics,
        )

    def create_source_steer(self):
        return ac.SteeringVector(
            steer_type='true level',
            ref=self.obs.pos.squeeze(),
            mics=self.noisy_mics,
            grid=ac.ImportGrid(),  # is filled later
            env=self.env,
        )

    def create_micgeom_sampler(self):
        return sp.MicGeomSampler(
            random_var=norm(loc=0, scale=0.001),
            ddir=np.array([[1.0], [1.0], [0]]),
            target=self.noisy_mics,
            mpos_init=self.mics.pos_total,
        )

    def create_location_sampler(self):
        ap = np.round(self.mics.aperture, decimals=12)
        z = self.grid.z
        location_sampler = sp.LocationSampler(
            random_var=(norm(0, 0.1688 * ap), norm(0, 0.1688 * ap), norm(z, 0)),
            x_bounds=(-0.5 * ap, 0.5 * ap),
            y_bounds=(-0.5 * ap, 0.5 * ap),
            z_bounds=(0.5 * ap, 0.5 * ap),
            nsources=self.max_nsources,
        )
        if self.snap_to_grid:
            location_sampler.grid = self.source_grid
        return location_sampler

    def create_rms_sampler(self):
        random_func = partial(sample_rms, self.max_nsources)
        return sp.ContainerSampler(random_func=random_func)

    def create_signal_seed_sampler(self):
        return sp.ContainerSampler(random_func=sample_signal_seed)

    def create_nsources_sampler(self):
        return sp.NumericAttributeSampler(
            random_var=poisson(mu=3, loc=1),
            attribute='nsources',
            equal_value=True,
            target=[self.location_sampler],
            filter=lambda x: (x <= self.max_nsources) and (x >= self.min_nsources),
        )

    def create_mic_noise_sampler(self):
        return sp.ContainerSampler(random_func=sample_mic_noise_variance)

    def create_signal_length_sampler(self):
        return sp.ContainerSampler(random_func=sample_signal_length)

    @staticmethod
    def _prepare_noise_params(sampler, prms_sq):
        noise_sampler = sampler.get(5)
        if noise_sampler is not None:
            noise_signal_ratio = noise_sampler.target  # normalized noise variance
            return prms_sq.sum() * noise_signal_ratio
        return None

    @staticmethod
    def _prepare_mics(sampler, mics):
        micgeom_sampler = sampler.get(1)
        if micgeom_sampler is None:
            return mics
        return micgeom_sampler.target

    @staticmethod
    def _prepare_source_params(sampler, fs):
        seed_sampler = sampler.get(2)
        rms_sampler = sampler.get(3)
        loc_sampler = sampler.get(4)
        signal_length_sampler = sampler.get(6)
        loc = loc_sampler.target
        rms_sq = rms_sampler.target[: loc.shape[1]] ** 2
        source_seeds = [seed_sampler.target + i for i in range(loc.shape[1])]
        num_samples = signal_length_sampler.target * fs if signal_length_sampler is not None else None
        return loc, rms_sq, source_seeds, num_samples

    @staticmethod
    def _prepare_signals_welch(prms_sq, sources, num_samples, source_seeds):
        signals = get_all_source_signals(sources)
        for i, signal in enumerate(signals):
            signal.seed = source_seeds[i]
            signal.rms = np.sqrt(prms_sq[i])
            if num_samples is not None:
                signal.num_samples = num_samples
        return signals

    @staticmethod
    def _prepare_sources_welch(sources, loc, mics):
        # set source locations
        subset_sources = sources[: loc.shape[1]]
        for i, src in enumerate(subset_sources):
            src.loc = (loc[0, i], loc[1, i], loc[2, i])  # apply wishart locations
            src.mics = mics
        return subset_sources

    @staticmethod
    def _prepare_spectra_welch(subset_sources, freq_data, fft_spectra, fft_obs_spectra, obs):
        freq_data.source.sources = subset_sources  # apply subset of sources
        fft_spectra.source = freq_data.source  # only for spectrogram feature
        # update observation point
        obs_sources = deepcopy(subset_sources)
        for src in obs_sources:
            src.mics = obs
        fft_obs_spectra.source = ac.SourceMixer(sources=obs_sources)

    @staticmethod
    def _prepare_noise_welch(sampler, prms_sq, seed, freq_data, num_samples, mics):
        noise_prms_sq = DatasetSyntheticConfig._prepare_noise_params(sampler, prms_sq)
        mic_noise = get_uncorrelated_noise_source_recursively(freq_data.source)
        if mic_noise:
            mic_noise_signal = mic_noise[0].signal
            mic_noise_signal.num_samples = num_samples
            if noise_prms_sq is not None:
                mic_noise_signal.rms = np.sqrt(noise_prms_sq)
                mic_noise_signal.seed = seed
                freq_data.source.source.mics = mics

    @staticmethod
    def _prepare_spectra_wishart(
        mics, freq_data, loc, prms_sq, source_seeds, noise_prms_sq, num_samples, custom_transfer=None
    ):
        nfft = freq_data.fftfreq().shape[0]
        if num_samples is not None:
            freq_data.num_samples = num_samples
        freq_data.steer.grid = ac.ImportGrid(pos=loc)  # set source locations
        freq_data.steer.mics = mics
        freq_data.seed = source_seeds[0]
        freq_data.Q = np.repeat(np.diag(prms_sq / nfft)[np.newaxis, :, :], nfft, axis=0)
        if noise_prms_sq is not None:
            sig_identity = np.eye(mics.num_mics)[np.newaxis, :, :] * (noise_prms_sq / nfft)
            freq_data.noise = np.repeat(sig_identity, nfft, axis=0)
        else:
            freq_data.noise = None
        freq_data.custom_transfer = custom_transfer

    @staticmethod
    def calc_welch_prepare_func(sampler, mics, beamformer, sources, source_steer, fft_spectra, fft_obs_spectra, obs):
        cf = DatasetSyntheticConfig
        freq_data = beamformer.freq_data
        mics = cf._prepare_mics(sampler, mics)
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        source_steer.grid = ac.ImportGrid(pos=loc)
        subset_sources = cf._prepare_sources_welch(sources, loc, mics)
        signals = cf._prepare_signals_welch(prms_sq * source_steer.r0**2, subset_sources, num_samples, source_seeds)
        num_samples = signals[0].num_samples
        cf._prepare_spectra_welch(subset_sources, freq_data, fft_spectra, fft_obs_spectra, obs)
        cf._prepare_noise_welch(sampler, prms_sq, source_seeds[0] + 1000, freq_data, num_samples, mics)
        return {
            'loc': loc,
            'prms_sq': prms_sq,
        }

    @staticmethod
    def calc_analytic_prepare_func(sampler, mics, freq_data):
        cf = DatasetSyntheticConfig
        mics = DatasetSyntheticConfig._prepare_mics(sampler, mics)
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        noise_prms_sq = DatasetSyntheticConfig._prepare_noise_params(sampler, prms_sq)
        # set Wishart simulator
        cf._prepare_spectra_wishart(
            mics,
            freq_data,
            loc,
            prms_sq,
            source_seeds,
            noise_prms_sq,
            num_samples,
        )
        return {
            'loc': loc,
            'prms_sq': prms_sq,
        }

    def get_prepare_func(self):
        if self.mode == 'welch':
            prepare_func = partial(
                self.calc_welch_prepare_func,
                mics=self.mics,
                beamformer=self.beamformer,
                sources=self.sources,
                source_steer=self.source_steer,
                fft_spectra=self.fft_spectra,
                fft_obs_spectra=self.fft_obs_spectra,
                obs=self.obs,
            )
        else:
            prepare_func = partial(self.calc_analytic_prepare_func, mics=self.mics, freq_data=self.beamformer.freq_data)
        return prepare_func

    def get_cleanup_func(self, features):
        def cleanup_func(sampler, data):
            # remove all items not in features
            keys_to_remove = [key for key in data if key not in features]
            for key in keys_to_remove:
                del data[key]
            return data

        return cleanup_func


class DatasetSyntheticISMConfig(DatasetSyntheticConfig):
    """Unsupported developer-only configuration for impulse-response-based synthetic scenes."""

    rt60 = Float(2.0, desc='reverberation time T60 in seconds')
    room_size = List([6, 4, 3], desc='room dimensions [x, y, z] in meters')

    def create_sources(self):
        sources = []
        for signal in self.signals:
            sources.append(
                ac.PointSourceConvolve(
                    signal=signal,
                    mics=self.noisy_mics,
                    env=self.env,
                    extend_signal=True,
                ),
            )
        return sources

    @staticmethod
    def _prepare_ir(mics, freq_data, loc, ref_loc, room_params, c, domain='frequency'):
        fftfreq = freq_data.fftfreq()
        nfft = freq_data.fftfreq().shape[0]
        nsources = loc.shape[1]
        num_mics = mics.num_mics

        # we don't use a chunk cache here, since we access the data only once
        # finding the SRIR matching the location
        if domain == 'frequency':
            transfer = np.empty((nfft, num_mics + 1, nsources), dtype=complex)

        rdim = room_params['room_size']
        rt60 = room_params['rt60']
        # calculate center of the area spanned by mics and sources
        ref_loc = np.atleast_2d(ref_loc).T
        all_pos = np.hstack((mics.pos_total, loc, ref_loc))
        center = 0.5 * (np.min(all_pos, axis=1) + np.max(all_pos, axis=1))[:, np.newaxis]
        # shift center to center of the room
        room_center = np.array([[rdim[0] / 2], [rdim[1] / 2], [rdim[2] / 2]]) - center
        # shift positions to center of the room
        mloc = np.hstack((mics.pos, ref_loc)) + room_center
        sloc = loc + room_center
        #: missing speed of sound
        irs = get_ir(freq_data.sample_freq, rdim, mloc, sloc, rt60)
        h_norm = np.zeros(nsources)
        # get longest ir length
        max_ir_len = max([irs[j][i].shape[0] for i in range(nsources) for j in range(num_mics)])
        # pad irs to same length
        irs_padded = np.zeros((num_mics + 1, nsources, max_ir_len))
        for i in range(nsources):
            h_norm[i] = np.sum(irs[-1][i] ** 2)
            for j in range(num_mics + 1):
                ir = irs[j][i]
                irs_padded[j, i, : ir.shape[0]] = ir
            if domain == 'frequency':
                transfer[:, :, i] = calc_transfer(
                    irs_padded[:, i, :], freq_data.sample_freq, freq_data.block_size, fftfreq
                )
                # normalize by ref norm
        if domain == 'frequency':
            transfer /= np.sqrt(h_norm[np.newaxis, np.newaxis, :])
            return transfer
        # normalize irs
        irs_padded /= np.sqrt(h_norm[np.newaxis, :, np.newaxis])
        return irs_padded

    @staticmethod
    def _prepare_ir_kernel(ir, sources, ref_sources):
        for i, src in enumerate(sources):
            src.kernel = ir[:-1, i, :].T
        for i, src in enumerate(ref_sources):
            src.kernel = ir[-1, i, :].T[:, np.newaxis]

    @staticmethod
    def calc_analytic_prepare_func(sampler, mics, freq_data, room_params):
        cf = DatasetSyntheticConfig
        cism = DatasetSyntheticISMConfig
        mics = cf._prepare_mics(sampler, mics)
        c = freq_data.steer.env.c
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        ref_loc = freq_data.steer.ref
        H = cism._prepare_ir(mics, freq_data, loc, ref_loc, room_params=room_params, c=c, domain='frequency')
        noise_prms_sq = cf._prepare_noise_params(sampler, prms_sq)
        cf._prepare_spectra_wishart(
            mics,
            freq_data,
            loc,
            prms_sq,
            source_seeds,
            noise_prms_sq,
            num_samples,
            custom_transfer=H[:, :-1, :],
        )
        return {
            'loc': loc,
            'prms_sq': prms_sq,
            'h_sq': np.real(H[:, -1, :] * H[:, -1, :].conj()),
        }

    @staticmethod
    def calc_welch_prepare_func(sampler, mics, beamformer, sources, fft_spectra, fft_obs_spectra, obs, room_params):
        cf = DatasetSyntheticConfig
        cism = DatasetSyntheticISMConfig
        freq_data = beamformer.freq_data
        fftfreq = freq_data.fftfreq()

        mics = cf._prepare_mics(sampler, mics)
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        c = sources[0].env.c
        ir = cism._prepare_ir(mics, freq_data, loc, obs.pos.squeeze(), room_params, domain='time', c=c)
        subset_sources = cf._prepare_sources_welch(sources, loc, mics)
        signals = cf._prepare_signals_welch(prms_sq, subset_sources, num_samples, source_seeds)
        num_samples = signals[0].num_samples
        cf._prepare_spectra_welch(subset_sources, freq_data, fft_spectra, fft_obs_spectra, obs)
        cf._prepare_noise_welch(sampler, prms_sq, source_seeds[0] + 1000, freq_data, num_samples, mics)
        cism._prepare_ir_kernel(ir, freq_data.source.sources, fft_obs_spectra.source.sources)
        # calc ref transfer for prms_sq_f
        H_ref = calc_transfer(ir[-1, :, :], freq_data.sample_freq, freq_data.block_size, fftfreq)
        return {
            'loc': loc,
            'prms_sq': prms_sq,
            'h_sq': np.real(H_ref * H_ref.conj()),
        }

    def get_prepare_func(self):
        room_params = {
            'room_size': self.room_size,
            'rt60': self.rt60,
        }
        if self.mode == 'welch':
            prepare_func = partial(
                self.calc_welch_prepare_func,
                mics=self.mics,
                beamformer=self.beamformer,
                sources=self.sources,
                fft_spectra=self.fft_spectra,
                fft_obs_spectra=self.fft_obs_spectra,
                obs=self.obs,
                room_params=room_params,
            )
        else:
            prepare_func = partial(
                self.calc_analytic_prepare_func,
                mics=self.mics,
                freq_data=self.beamformer.freq_data,
                room_params=room_params,
            )
        return prepare_func


class DatasetSyntheticISM(DatasetSynthetic):
    """Unsupported developer-only dataset class for impulse-response-based synthetic scenes."""

    def __init__(
        self,
        mode='welch',
        mic_pos_noise=True,
        mic_sig_noise=True,
        snap_to_grid=False,
        random_signal_length=False,
        signal_length=5,
        fs=13720.0,
        min_nsources=1,
        max_nsources=10,
        rt60=2.0,
        tasks=1,
        remote_args=None,
        logger=None,
        config=None,
    ):
        """
        Parameters
        ----------
        config : DatasetSyntheticISMConfig
            Configuration object. Defaults to None. If None, a default configuration
            object is created.
        kwargs : dict
            Additional keyword arguments passed to the DatasetSynthetic constructor.
        """
        require_ir_support()
        if config is None:
            config = DatasetSyntheticISMConfig(
                mode=mode,
                signal_length=signal_length,
                fs=fs,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                mic_pos_noise=mic_pos_noise,
                mic_sig_noise=mic_sig_noise,
                snap_to_grid=snap_to_grid,
                random_signal_length=random_signal_length,
                rt60=rt60,
            )
        super().__init__(config=config, tasks=tasks, remote_args=remote_args, logger=logger)
