
from copy import deepcopy
from pathlib import Path

import acoular as ac
import h5py as h5
import numpy as np
from traits.api import Bool, DelegatesTo, Either, Enum, Float, HasTraits, Instance, Int, List, Property, Str, cached_property, observe

import acoupipe.sampler as sp
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.utils import get_all_source_signals


class MsmSetupBase(HasTraits):
    """Base class defining the virtual measurement setup.

    The objects are created lazily, i.e. only when they are accessed.
    """

    env = Instance(ac.Environment, args=(), desc="environment configuration")
    mics = Instance(ac.MicGeom, args=(), desc="microphone geometry configuration")
    grid = Instance(ac.Grid, args=(), desc="grid configuration")
    source_grid = Either(None, Instance(ac.Grid, desc="source grid configuration. Default is None."))
    beamformer = Instance(ac.BeamformerBase, args=(), desc="beamformer configuration")
    freq_data = Instance(ac.PowerSpectra, args=(), desc="frequency domain data configuration")
    steer = Instance(ac.SteeringVector, args=(), desc="steering vector configuration")

    def build(self):
        """Connect the related objects."""
        self.steer.env = self.env
        self.steer.grid = self.grid
        self.steer.mics = self.mics
        self.beamformer.steer = self.steer
        self.beamformer.freq_data = self.freq_data


class MIRACLESetup(MsmSetupBase):

    filename = Either(Instance(Path), Str, None)
    env = Property(depends_on="filename")
    mics = Property(depends_on="filename")

    @cached_property
    def _get_mics(self):
        with h5.File(self.filename, "r") as file:
            mpos_tot = file["data/location/receiver"][()].T
        return ac.MicGeom(mpos_tot = mpos_tot)

    @cached_property
    def create_env(self):
        with h5.File(self.filename, "r") as file:
            c = np.mean(file["metadata/c0"][()])
        return ac.Environment(c=c)


class SyntheticSetup(MsmSetupBase):
    signal_length = Float(desc="signal length in seconds")
    fs = DelegatesTo("freq_data", prefix="sample_freq", desc="sampling frequency")
    c = DelegatesTo("env", prefix="c", desc="Speed of sound in m/s.")
    ref = DelegatesTo("steer", prefix="ref", desc="reference point for steering vector calculation")
    block_size = DelegatesTo("freq_data", prefix="block_size", desc="block size for fft")
    overlap = DelegatesTo("freq_data", prefix="overlap", desc="overlap for fft")


class SyntheticSetupAnalytic(SyntheticSetup):

    freq_data = Instance(PowerSpectraAnalytic, args=(), desc="frequency domain data configuration")
    mode = DelegatesTo("freq_data", prefix="mode", desc="mode for frequency domain data generation")

    @observe(["freq_data.numsamples"])
    def _set_signal_length(self, event):
        self.signal_length = int(self.freq_data.numsamples/self.freq_data.sample_freq)

    @observe(["signal_length", "fs"])
    def _set_numsamples(self, event):
        self.freq_data.numsamples = int(self.fs*self.signal_length)

    def build(self):
        """Connect the related objects."""
        self.steer.env = self.env
        self.steer.grid = self.grid
        self.steer.mics = self.mics
        #self.freq_data.steer = deepcopy(self.steer)
        self.beamformer.steer = self.steer
        self.beamformer.freq_data = self.freq_data


class SyntheticSetupWelch(SyntheticSetup):

    time_data = Instance(ac.SourceMixer, args=(), desc="object holding the time domain data")
    sources = List(Instance(ac.PointSource), desc="list of point sources")
    noise = Instance(ac.UncorrelatedNoiseSource, desc="noise source configuration")
    fs = Property(desc="sampling frequency")
    window = DelegatesTo("freq_data", prefix="window", desc="window function for fft")
    mode = Enum("welch", desc="mode for frequency domain data generation")
    _fs = Float(desc="sampling frequency")

    def build(self):
        """Connect the related objects."""
        self.time_data.sources = self.sources + [self.noise]
        self.steer.env = self.env
        self.steer.grid = self.grid
        self.steer.mics = self.mics
        self.freq_data.source = self.time_data
        self.beamformer.steer = self.steer
        self.beamformer.freq_data = self.freq_data

    @observe(["freq_data.source.numsamples"])
    def _set_signal_length(self, event):
        self.signal_length = int(self.freq_data.source.numsamples/self.freq_data.sample_freq)

    @observe(["signal_length", "fs"])
    def _set_numsamples(self, event):
        signals = get_all_source_signals(source_list=self.sources + [self.noise])
        for signal in signals:
            signal.numsamples = int(self.fs*self.signal_length)

    def _set_fs(self, fs):
        self._fs = fs
        signals = get_all_source_signals(source_list=self.sources + [self.noise])
        for signal in signals:
            signal.sample_freq = self._fs

    def _get_fs(self):
        signals = get_all_source_signals(source_list=self.sources + [self.noise])
        all_fs = [signal.sample_freq for signal in signals]
        if len(set(all_fs)) == 1:
            fs = all_fs[0]
            if self._fs != fs:
                self._fs = fs
        else:
            raise ValueError("Sampling frequency of all signals has to be the same.")
        return self._fs



class SamplerSetupBase(HasTraits):

    msm_setup = Instance(MsmSetupBase, desc="msm setup configuration")
    numsampler = Property()

    def _get_numsampler(self):
        return 0

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.sampler.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The key defines the idx in the sample order.

        e.g.:
        >>> sampler = {
        >>>     0 : BaseSampler(...),
        >>>     1 : BaseSampler(...),
        >>>     ...
        >>> }

        Returns
        -------
        dict
            dictionary containing the sampler objects
        """
        sampler = {}
        return sampler


class SyntheticSamplerSetup(SamplerSetupBase):
    """The sampler setup for synthetic data generation.

    Parameters
    ----------
    min_nsources : int
        Minimum number of sources. Default is 1.
    max_nsources : int
        Maximum number of sources. Default is 10.
    mic_pos_noise : bool (default: True)
        Apply positional noise to microphone geometry.
    mic_sig_noise : bool
        Apply signal noise to microphone signals. Default is :code:`True`.
    snap_to_grid : bool
        Snap sampled source locations to source grid. Default is :code:`False`.
    random_signal_length : bool
        Randomize signal length (uniformly sampled signal length [1s,10s]). Default is :code:`False`.
    nsources_random_var : scipy.stats.rv_continuous
        Random variable for the number of sources sampler. Default is Poisson distribution.
    location_random_var : tuple of scipy.stats.rv_continuous
        Random variable for the location sampler. Default is normal bivariate distribution.
    mic_pos_random_var : scipy.stats.rv_continuous
        Random variable for the microphone geometry noise sampler. Default is normal distribution.
    mic_sig_random_var : callable
        Random variable for the microphone signal noise sampler. Default is :func:`sample_mic_noise_variance`.
    strenght_random_var : callable
        Random variable for the source strength sampler. Default is :func:`sample_rms`.
    signal_length_random_var : callable
        Random variable for the signal length sampler. Default is :func:`sample_signal_length`.
    """

    min_nsources = Either(None, Int, desc="minimum number of sources")
    max_nsources = Either(None, Int, desc="maximum number of sources")
    # positional noise to microphone geometry
    mic_pos_noise = Bool(True, desc="apply positional noise to microphone arrangement")
    mic_pos_sampler = Instance(sp.MicGeomSampler, desc="sampler for microphone geometry noise")
    mic_pos_random_var = DelegatesTo("mic_pos_sampler", prefix="random_var", desc="random variable for the microphone geometry noise sampler")
    mic_pos_ddir = DelegatesTo("mic_pos_sampler", prefix="ddir", desc="deviation direction for the microphone geometry noise sampler")
    # source location sampling
    loc_sampler = Instance(sp.LocationSampler, desc="sampler for source location")
    loc_random_var = DelegatesTo("loc_sampler", prefix="random_var", desc="random variable for the location sampler")
    snap_to_grid = Bool(False, desc="snap source locations to source_grid of measurement setup")
    # number of sources sampling
    nsources_sampler = Instance(sp.NumericAttributeSampler, desc="sampler for the number of sources")
    nsources_random_var = DelegatesTo("nsources_sampler", prefix="random_var", desc="random variable for the number of sources sampler")
    # source strength sampling
    strength_sampler = Instance(sp.ContainerSampler, desc="sampler for the source strength")
    strength_random_func = DelegatesTo("strength_sampler", prefix="random_func", desc="random variable for the source strength sampler")
    # microphone noise variance sampling
    mic_sig_noise = Bool(True, desc="apply signal noise to microphone signals")
    mic_sig_noise_sampler = Instance(sp.ContainerSampler, desc="sampler for the microphone noise variance")
    mic_sig_noise_random_func = DelegatesTo("mic_sig_noise_sampler", prefix="random_func", desc="random variable for the microphone noise variance sampler")
    # signal length sampling
    random_signal_length = Bool(False, desc="randomize signal length")
    signal_length_sampler = Instance(sp.ContainerSampler, desc="sampler for the signal length")
    signal_length_random_func = DelegatesTo("signal_length_sampler", prefix="random_func", desc="random variable for the signal length sampler")
    # seed sampling
    seed_sampler = Instance(sp.ContainerSampler, desc="sampler for the signal seed")
    seed_random_func = DelegatesTo("seed_sampler", prefix="random_func", desc="random variable for the signal seed sampler")

    def _get_numsampler(self):
        return len(self.get_sampler())

    @observe(["msm_setup.mics","mic_pos_sampler"])
    def _set_mics(self, event):
        if self.mic_pos_sampler is not None:
            self.mic_pos_sampler.mpos_init = self.msm_setup.mics.mpos_tot
            self.mic_pos_sampler.target = deepcopy(self.msm_setup.mics)

    @observe(["min_nsources","max_nsources","loc_sampler"])
    def _set_nsources(self, event):
        if (self.loc_sampler is not None) and (self.max_nsources is not None):
            self.loc_sampler.nsources = self.max_nsources
        if self.nsources_sampler is not None:
            if self.max_nsources and self.min_nsources:
                self.nsources_sampler.filter = lambda x: (x <= self.max_nsources) and (x >= self.min_nsources)
            elif self.max_nsources:
                self.nsources_sampler.filter = lambda x: x <= self.max_nsources
            elif self.min_nsources:
                self.nsources_sampler.filter = lambda x: x >= self.min_nsources
            else:
                self.nsources_sampler.filter = None

    @observe(["snap_to_grid","msm_setup.source_grid","msm_setup.grid","loc_sampler"])
    def _set_snap_to_grid(self, event):
        if self.snap_to_grid:
            if self.msm_setup.source_grid is None:
                grid = self.msm_setup.grid
            else:
                grid = self.msm_setup.source_grid
            if self.loc_sampler is not None:
                self.loc_sampler.grid = grid

    @observe(["loc_sampler","nsources_sampler"])
    def _set_target(self, event):
        if self.nsources_sampler is not None:
            self.nsources_sampler.target = [self.loc_sampler]

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.sampler.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The key defines the idx in the sample order.

        e.g.:
        >>> sampler = {
        >>>     0 : BaseSampler(...),
        >>>     1 : BaseSampler(...),
        >>>     ...
        >>> }

        Returns
        -------
        dict
            dictionary containing the sampler objects
        """
        sampler = {
            2 : self.seed_sampler,
            3 : self.strength_sampler,
            4 : self.loc_sampler,
            }
        if self.max_nsources != self.min_nsources:
            sampler[0] = self.nsources_sampler
        if self.mic_pos_noise:
            sampler[1] = self.mic_pos_sampler
        if self.mic_sig_noise:
            sampler[5] = self.mic_sig_noise_sampler
        if self.random_signal_length:
            sampler[6] = self.signal_length_sampler
        return sampler


class ISMSamplerSetup(SyntheticSamplerSetup):

    room_size_sampler = Instance(sp.ContainerSampler, desc="sampler for the room size")
    absoption_coeff_sampler = Instance(sp.ContainerSampler, desc="samples pyroomacoustics rooms (e.g. ShoeBox rooms)")
    room_placement_sampler = Instance(sp.ContainerSampler,
        desc="the relative position of the source-microphone array center in the room")

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.sampler.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The key defines the idx in the sample order.

        e.g.:
        >>> sampler = {
        >>>     0 : BaseSampler(...),
        >>>     1 : BaseSampler(...),
        >>>     ...
        >>> }

        Returns
        -------
        dict
            dictionary containing the sampler objects
        """
        sampler = {
            2 : self.seed_sampler,
            3 : self.strength_sampler,
            4 : self.loc_sampler,
            7 : self.room_size_sampler,
            8 : self.absoption_coeff_sampler,
            9 : self.room_placement_sampler,
            }
        if self.max_nsources != self.min_nsources:
            sampler[0] = self.nsources_sampler
        if self.mic_pos_noise:
            sampler[1] = self.mic_pos_sampler
        if self.mic_sig_noise:
            sampler[5] = self.mic_sig_noise_sampler
        if self.random_signal_length:
            sampler[6] = self.signal_length_sampler
        return sampler
