
from copy import deepcopy
from functools import partial

import acoular as ac
import numpy as np
from scipy.stats import norm, poisson
from traits.api import (
    Any,
    Bool,
    CArray,
    Enum,
    Float,
    HasPrivateTraits,
    HasTraits,
    Instance,
    Int,
    Property,
    ReadOnly,
    Str,
    cached_property,
    observe,
)

import acoupipe.sampler as sp
from acoupipe.datasets.micgeom import tub_vogel64_ap1
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic


def create_default_grid():
    return ac.RectGrid(y_min=-0.5, y_max=0.5, x_min=-0.5, x_max=0.5,z=0.5, increment=1/63)

class MsmSetupBase(HasPrivateTraits):
    """Base class defining the virtual measurement setup.

    The objects are created lazily, i.e. only when they are accessed.
    """

    env = Instance(ac.Environment, args=(), desc="environment configuration") # created lazy!
    mics = Instance(ac.MicGeom, kw={"mpos_tot":tub_vogel64_ap1}, desc="microphone geometry configuration") # created lazy!
    grid = Instance(
        ac.Grid,factory=create_default_grid, desc="grid configuration")
    source_grid = Instance(ac.Grid, desc="source grid configuration. Default is None.")
    beamformer = Property(depends_on=["freq_data","steer"], desc="beamformer configuration")
    freq_data = Property()
    steer = Property()

    #TODO: might it be better to put all this in the init of the config to set the default -> not really needed ??

    # fast access attributes (can be used to set the values at multiple places without the need to access the object)
    min_nsources = Int(1, desc="minimum number of sources")
    max_nsources = Int(10, desc="maximum number of sources")
    signal_length = Float(5.0, desc="signal length in seconds")
    ref = CArray(dtype=float, shape=(3,), value=tub_vogel64_ap1[:,63], desc="reference point for steering vector calculation")
    steer_type = Str("true level", desc="steering vector type")
    fs = Int(13720, desc="sampling frequency")
    block_size = Int(128, desc="block size for fft")
    overlap = Str("50%", desc="overlap for fft")

    _beamformer = Instance(ac.BeamformerBase,
        kw={"precision":"float32", "cached":False, "r_diag":False}, desc="beamformer configuration")
    _freq_data = Instance(ac.PowerSpectra(cached=False, precision="complex64"), desc="frequency domain data configuration")
    _steer = Instance(ac.SteeringVector(), desc="steering vector configuration")

    def _get_beamformer(self):
        return self._beamformer

    def _set_beamformer(self, beamformer):
        self._beamformer = beamformer

    def _get_freq_data(self):
        return self._freq_data

    def _get_steer(self):
        return self._steer

    @observe(["block_size","overlap"])
    def _set_fft_params(self, event):
        print("set fft params")
        self.freq_data.block_size = self.block_size
        self.freq_data.overlap = self.overlap

    @observe(["grid","env", "mics","steer_type", "ref"])
    def _set_steer(self, event):
        print("set steer")
        self.steer.ref = self.ref
        self.steer.grid = self.grid
        self.steer.env = self.env
        self.steer.mics = self.mics
        self.steer.steer_type = self.steer_type

    @observe(["_beamformer"])
    def _set_beamformer(self, event):
        print("set beamformer")
        self.beamformer.freq_data = self.freq_data
        self.beamformer.steer = self.steer


class SyntheticSetupWelch(MsmSetupBase):

    sources = Property()
    mic_sig_noise_source = Property()
    mic_sig_noise = Bool(True, desc="apply signal noise to microphone signals")
    fft_spectra =  ReadOnly(ac.FFTSpectra(precision="complex64"),
        desc="FFT spectra configuration (only for spectrogram feature)")
    window = Str("Rectangular", desc="window function for fft")
    _sources = Any()
    _mic_sig_noise_source = Instance(ac.UncorrelatedNoiseSource, desc="noise source configuration")

    @observe(["signal_length","fs"])
    def _set_numsamples(self, event):
        for source in self.sources:
            source.signal.numsamples = int(self.fs*self.signal_length)
        if self.mic_sig_noise:
            self.mic_sig_noise_source.signal.numsamples = int(self.fs*self.signal_length)

    @observe(["grid","env", "mics","steer_type", "ref"])
    def _set_steer(self, event):
        self.steer.ref = self.ref
        self.steer.grid = self.grid
        self.steer.env = self.env
        self.steer.mics = self.mics
        self.steer.steer_type = self.steer_type

    @observe(["block_size","overlap", "window"])
    def _set_fft_params(self, event):
        self.freq_data.block_size = self.block_size
        self.freq_data.overlap = self.overlap
        self.freq_data.window = self.window
        self.fft_spectra.block_size = self.block_size
        self.fft_spectra.overlap = self.overlap
        self.fft_spectra.window = self.window

    def _set_sources(self, sources):
        self._sources = sources

    def _set_mic_sig_noise_source(self, mic_noise_source):
        self._mic_noise_source = mic_noise_source

    @cached_property
    def _get_mic_sig_noise_source(self):
        if not self._mic_noise_source and self.mic_sig_noise:
            self._mic_noise_source = ac.UncorrelatedNoiseSource(
                signal = ac.WNoiseGenerator(
                    seed = 1000,
                    sample_freq = self.fs,
                    numsamples = int(self.fs*self.signal_length),
                    ),
                mics = self.mics
                )
        return self._mic_noise_source

    @cached_property
    def _get_sources(self):
        if not self._sources:
            sources = []
            for i in range(self.max_nsources):
                sources.append(
                    ac.PointSource(signal = ac.WNoiseGenerator(
                            seed = i+1,
                            sample_freq = self.fs,
                            numsamples = int(self.fs*self.signal_length),
                            ),
                            env = self.env,
                            mics = self.mics,
                            )
                    )
            if self.mic_sig_noise:
                sources.append(self.mic_sig_noise_source)
            self._sources = sources
        return self._sources


class SyntheticSetupAnalytic(MsmSetupBase):

    freq_data = ReadOnly(PowerSpectraAnalytic(precision="complex64"), desc="frequency domain data configuration")
    mode = Enum("wishart", "analytic", default="wishart", desc="mode for frequency domain data generation")

    @observe("mode")
    def _set_mode(self, event):
        self.freq_data.mode = self.mode

    @observe(["signal_length","fs"])
    def _set_numsamples(self, event):
        self.freq_data.numsamples = int(self.fs*self.signal_length)

    @observe(["grid","env", "mics","steer_type", "ref"])
    def _set_steer(self, event):
        self.steer.ref = self.ref
        self.steer.grid = self.grid
        self.steer.env = self.env
        self.steer.mics = self.mics
        self.steer.steer_type = self.steer_type

    @observe(["block_size","overlap","fs"])
    def _set_fft_params(self, event):
        self.freq_data.block_size = self.block_size
        self.freq_data.overlap = self.overlap
        self.freq_data.sample_freq = self.fs


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

def sample_mic_noise_variance(rng):
    """Draw microphone noise variance, uniform distribution."""
    return rng.uniform(10e-6,0.1)

def sample_rms(nsources, rng):
    """Draw sources' squared rms pressures from Rayleigh distribution."""
    return np.sqrt(rng.rayleigh(5,nsources))

def sample_signal_length(rng):
    return rng.uniform(1,10)

class SyntheticSamplerSetup(SamplerSetupBase):
    #TODO: can the sampler be created in the __init__ method? -> deepcopy in get_sampler?
    # this would enable the user to make changes to the sampler after the setup has been created
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

    mic_pos_noise = Bool(True, desc="apply positional noise to microphone geometry")
    mic_sig_noise = Bool(True, desc="apply signal noise to microphone signals")
    snap_to_grid = Bool(False, desc="snap source locations to grid of measurement setup")
    random_signal_length = Bool(False, desc="randomize signal length")
    nsources_random_var = Any(poisson(mu=3, loc=1), desc="random variable for the number of sources sampler")
    location_random_var = Any((norm(0,0.1688),norm(0,0.1688),norm(0.5,0)), desc="random variable for the location sampler")
    mic_pos_random_var = Any(norm(loc=0, scale=0.001), desc="random variable for the microphone geometry noise sampler")
    mic_sig_random_var = Any(sample_mic_noise_variance, desc="callable random variable for the microphone signal noise sampler")
    strenght_random_var = Any(sample_rms, desc="callable random variable for the source strength sampler")
    signal_length_random_var = Any(sample_signal_length, desc="callable random variable for the signal length sampler")

    def _get_numsampler(self):
        numsampler = 3
        if self.mic_pos_noise:
            numsampler += 1
        if self.mic_sig_noise:
            numsampler += 1
        if self.random_signal_length:
            numsampler += 1
        if self.msm_setup.max_nsources != self.msm_setup.min_nsources:
            numsampler += 1
        return numsampler

    def _create_signal_length_sampler(self):
        return sp.ContainerSampler(
            random_func = self.signal_length_random_var)

    def _create_mic_noise_sampler(self):
        return sp.ContainerSampler(
            random_func = self.mic_sig_random_var)

    def _create_micgeom_sampler(self):
        noisy_mics = deepcopy(self.msm_setup.mics)
        return sp.MicGeomSampler(
            random_var = self.mic_pos_random_var,
            ddir = np.array([[1.0], [1.0], [0]]),
            target = noisy_mics, # TODO: noisy mics public?
            mpos_init = self.msm_setup.mics.mpos_tot,)

    def _create_nsources_sampler(self, target):
        return sp.NumericAttributeSampler(
            random_var = self.nsources_random_var,
            attribute = "nsources",
            equal_value = True,
            target=target,
            filter=lambda x: (x <= self.msm_setup.max_nsources) and (
                x >= self.msm_setup.min_nsources))

    def _create_rms_sampler(self):
        random_func = partial(self.strenght_random_var, self.msm_setup.max_nsources)
        return sp.ContainerSampler(random_func = random_func)

    def _create_signal_seed_sampler(self):
        def sample_signal_seed(rng):
            return int(rng.uniform(1,1e9))
        return sp.ContainerSampler(
            random_func = sample_signal_seed)

    def _create_location_sampler(self):
        if self.msm_setup.source_grid is None:
            grid = self.msm_setup.grid
        else:
            grid = self.msm_setup.source_grid

        location_sampler = sp.LocationSampler(
            random_var = self.location_random_var,
            x_bounds = (grid.gpos[0].min(),grid.gpos[0].max()), #TODO: automatically set bounds from grid
            y_bounds = (grid.gpos[1].min(),grid.gpos[1].max()),
            z_bounds = (grid.gpos[2].min(),grid.gpos[2].max()),
            nsources = self.msm_setup.max_nsources,
            )
        if self.snap_to_grid:
            location_sampler.grid = grid
        return location_sampler

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
        location_sampler = self._create_location_sampler()
        sampler = {
            2 : self._create_signal_seed_sampler(),
            3 : self._create_rms_sampler(),
            4 : location_sampler,
            }
        if self.msm_setup.max_nsources != self.msm_setup.min_nsources:
            sampler[0] = self._create_nsources_sampler(
                target=[location_sampler])
        if self.mic_pos_noise:
            sampler[1] = self._create_micgeom_sampler()
        if self.mic_sig_noise:
            sampler[5] = self._create_mic_noise_sampler()
        if self.random_signal_length:
            sampler[6] = self._create_signal_length_sampler()
        return sampler



if __name__ == "__main__":

    import numpy as np

    from acoupipe.pipeline import BasePipeline, DistributedPipeline

    s = MsmSetupBase()
    p = BasePipeline(numsamples=1, sampler={0: np.random.RandomState(1)}, features=(lambda sampler, s: {"setup" : s}, s))
    res = next(p.get_data())
    print(res["setup"], s)

    # p2 = DistributedPipeline(numworkers=2, numsamples=1, sampler={0: np.random.RandomState(1)}, features=lambda sampler: {"setup" : MsmSetupBase()})
    # res = next(p2.get_data())

    s = MsmSetupBase()
    s._rtest = 2
    #import ray
    #s_ref = ray.put(s) # serialize the object
    s_ref = s
    # The object must be explicitly freed by calling ray._private.internal_api.free(obj_ref). !!!!

    func = (lambda sampler, s: {"setup" : id(s)}, s_ref)
    #func = partial(lambda sampler, s: {"setup" : s}, s_ref)
    p3 = DistributedPipeline(numworkers=6, numsamples=10, sampler={0: np.random.RandomState(1)},
                features=func)

    # measure time
    import time
    test_value = []
    for i, res in enumerate(p3.get_data()):
        test_value.append(res["setup"])
        if i == p3.numworkers:
            start = time.time()
    print(time.time()-start)
    print(set(test_value))

    # res = next(p3.get_data())
    # print(res["setup"], s)

#    ray._private.internal_api.free(s_ref)
