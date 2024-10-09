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
from traits.api import DelegatesTo, Instance

import acoupipe.sampler as sp
from acoupipe.datasets.base import ConfigBase, DatasetBase
from acoupipe.datasets.collection import FeatureCollectionBuilder
from acoupipe.datasets.features import (
    CSMDiagonalAnalytic,
    CSMDiagonalWelch,
    CSMFeature,
    CSMtriuFeature,
    EigmodeFeature,
    Feature,
    FloatFeature,
    IntFeature,
    SourcemapFeature,
    SpectrogramFeature,
    TargetmapFeature,
    TimeDataFeature,
)
from acoupipe.datasets.micgeom import tub_vogel64_ap1
from acoupipe.datasets.setup import SyntheticSamplerSetup, SyntheticSetup, SyntheticSetupAnalytic, SyntheticSetupWelch
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.transfer import TransferMonopole


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

    def __init__(self, config=None, tasks=1, logger=None, **kwargs):
        """Initialize the DatasetSynthetic object.

        Most of the input parameters are passed to the DatasetSyntheticConfig object, which creates
        all necessary objects for the simulation of microphone array data.

        Parameters
        ----------
        config : DatasetSyntheticConfig
            Configuration object. Defaults to None. If None, a default configuration
            object is created and the input parameters / kwargs are passed to it.
        tasks : int
            Number of parallel tasks. Defaults to 1.
        logger : logging.Logger
            Logger object. Defaults to None.
        **kwargs
            Additional keyword arguments passed to the DatasetSyntheticConfig object.
            * mode : str (default: 'welch')
                Type of CSM calculation method.
            * signal_length : float (default: 5)
                Length of the source signals in seconds.
            * fs : float (default: 13720)
                Sampling frequency in Hz.
            * min_nsources : int (default: 1)
                Minimum number of sources.
            * max_nsources : int (default: 10)
                Maximum number of sources.
            * mic_pos_noise : bool (default: True)
                Apply positional noise to microphone geometry.
            * mic_sig_noise : bool (default: True)
                Apply signal noise to microphone signals.
            * snap_to_grid : bool (default: False)
                Snap source locations to grid.
            * random_signal_length : bool (default: False)
                Randomize signal length.
            * fft_params
                FFT parameters with default items :code:`block_size=128`,
                :code:`overlap="50%"`, :code:`window="Hanning"` and :code:`precision="complex64"`.
            * env : ac.Environment
                Instance of acoular.Environment defining the environmental coditions,
                i.e. the speed of sound.
            * mics : ac.MicGeom
                Instance of acoular.MicGeom defining the microphone array geometry.
            * grid : ac.RectGrid
                Instance of acoular.RectGrid defining the grid on which the Beamformer calculates
            ... and all other parameters of the DatasetSyntheticConfig object.
        """
        if config is None:
            if kwargs.get("mode") in ["analytic", "wishart"]:
                config = DatasetSyntheticConfigAnalytic(**kwargs)
            else:
                config = DatasetSyntheticConfigWelch(**kwargs)
        super().__init__(config=config, tasks=tasks, logger=logger)



class DatasetSyntheticConfig(ConfigBase):

    msm_setup = Instance(SyntheticSetup, desc="SyntheticSetupAnalytic object.")
    sampler_setup = Instance(SyntheticSamplerSetup, desc="SyntheticSamplerSetup object.")

    def __init__(self, msm_setup=None, sampler_setup=None, **kwargs):

        if msm_setup is None: #set the default
            msm_setup = self._get_default_msm_setup(**kwargs)
        msm_setup.build()

        # build sampler setup
        if sampler_setup is None:
            sampler_setup = self.get_default_sampler_setup(msm_setup, **kwargs)

        # handle unrecognized kwargs
        all_keys = list(set(
            sampler_setup.__class__.class_traits().keys()) | set(
            msm_setup.__class__.class_traits().keys()))
        unrecognized_keys = set(kwargs.keys()) - set(all_keys)
        if unrecognized_keys:
            msg = f"Unrecognized keyword arguments: {unrecognized_keys} in {self.__class__.__name__}"
            raise ValueError(msg)
        super().__init__(msm_setup=msm_setup, sampler_setup=sampler_setup)

    def get_default_mic_sampler(self, msm_setup, **kwargs):
        if kwargs.get("mic_pos_sampler") is None:
            kwargs["mic_pos_sampler"] = sp.MicGeomSampler(
                    random_var = norm(loc=0, scale=0.001),
                    ddir = np.array([[1.0], [1.0], [0]]),
                    target = deepcopy(msm_setup.mics),
                    mpos_init = msm_setup.mics.mpos_tot,
                )
        return kwargs

    def get_default_loc_sampler(self, msm_setup, **kwargs):
        if kwargs.get("loc_sampler") is None:
            ap = msm_setup.mics.aperture
            kwargs["loc_sampler"] = sp.LocationSampler(
                random_var = (norm(0,0.1688*ap),norm(0,0.1688*ap),norm(0.5*ap,0)),
                x_bounds = (-0.5*ap,0.5*ap),
                y_bounds = (-0.5*ap,0.5*ap),
                z_bounds = (0.5*ap,0.5*ap),
                )
        return kwargs

    def get_default_nsources_sampler(self, msm_setup, **kwargs):
        if kwargs.get("nsources_sampler") is None:
            kwargs["nsources_sampler"] = sp.NumericAttributeSampler(
                random_var = poisson(mu=3, loc=1),
                attribute = "nsources",
                equal_value = True,
                target = [kwargs["loc_sampler"]],
                )
        return kwargs

    def get_default_strength_sampler(self, msm_setup, **kwargs):
        if kwargs.get("strength_sampler") is None:
            kwargs["strength_sampler"] = sp.ContainerSampler(
                random_func = lambda rng: np.sqrt(rng.rayleigh(5,kwargs["max_nsources"]))
            )
        return kwargs

    def get_default_mic_sig_noise_sampler(self, msm_setup, **kwargs):
        if kwargs.get("mic_sig_noise_sampler") is None:
            kwargs["mic_sig_noise_sampler"] = sp.ContainerSampler(
                random_func = lambda rng: rng.uniform(10e-6,0.1)
            )
        return kwargs

    def get_default_signal_length_sampler(self, msm_setup, **kwargs):
        if kwargs.get("signal_length_sampler") is None:
            kwargs["signal_length_sampler"] = sp.ContainerSampler(
                random_func = lambda rng: rng.uniform(1, 10), # between 1 and 10 seconds
            )
        return kwargs

    def get_default_seed_sampler(self, msm_setup, **kwargs):
        if kwargs.get("seed_sampler") is None:
            kwargs["seed_sampler"] = sp.ContainerSampler(
                random_func = lambda rng: int(rng.uniform(1,1e9))
            )
        return kwargs

    def get_default_mics(self, **kwargs):
        if kwargs.get("mics") is None:
            kwargs["mics"] = ac.MicGeom(mpos_tot=tub_vogel64_ap1)
        return kwargs

    def get_default_env(self, **kwargs):
        if kwargs.get("env") is None:
            kwargs["env"] = ac.Environment()
        return kwargs

    def get_default_grid(self, **kwargs):
        if kwargs.get("grid") is None:
            ap = kwargs["mics"].aperture
            kwargs["grid"] = ac.RectGrid(
                y_min=-0.5*ap, y_max=0.5*ap, x_min=-0.5*ap, x_max=0.5*ap, z=0.5*ap, increment=1/63*ap)
        return kwargs

    def get_default_steer(self, **kwargs):
        if kwargs.get("steer") is None:
            steer = ac.SteeringVector(ref=tub_vogel64_ap1[:,63])
            kwargs.update({"steer": steer})
        return kwargs

    def get_default_beamformer(self, **kwargs):
        if kwargs.get("beamformer") is None:
            kwargs["beamformer"] = ac.BeamformerBase(
                precision="float32", cached=False, r_diag=False, steer=kwargs["steer"])
        return kwargs

    def get_default_sampler_setup(self, msm_setup, **kwargs):
        kwargs["max_nsources"] = kwargs.get("max_nsources", 10)
        kwargs["min_nsources"] = kwargs.get("min_nsources", 1)
        kwargs = self.get_default_mic_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_loc_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_nsources_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_strength_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_mic_sig_noise_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_signal_length_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_seed_sampler(msm_setup, **kwargs)

        setup_class_attr = SyntheticSamplerSetup.class_traits().keys()
        sampler_setup = SyntheticSamplerSetup(msm_setup=msm_setup,
            **{key: kwargs.pop(key) for key in setup_class_attr if key in kwargs}
        )
        return sampler_setup

    def _get_time_data_feature(self):
        return TimeDataFeature(
            time_data=self.msm_setup.time_data,
            shape=(None, self.msm_setup.mics.num_mics)
        )

    def _get_spectrogram_feature(self, f, num):
        _fft_spectra = ac.FFTSpectra(
            window = self.msm_setup.freq_data.window,
            block_size = self.msm_setup.freq_data.block_size,
            overlap = self.msm_setup.freq_data.overlap,
            source = self.msm_setup.time_data
        )
        return SpectrogramFeature(
            freq_data=_fft_spectra, f=f, num=num,
            shape=(None, self._get_fdim(f), self.msm_setup.mics.num_mics)
        )

    def _get_csm_feature(self, f, num):
        return CSMFeature(
            freq_data = self.msm_setup.freq_data, f=f, num=num,
            shape = (self._get_fdim(f), self.msm_setup.mics.num_mics, self.msm_setup.mics.num_mics)
        )

    def _get_csmtriu_feature(self, f, num):
        return CSMtriuFeature(
            freq_data = self.msm_setup.freq_data, f=f, num=num,
            shape = (self._get_fdim(f), self.msm_setup.mics.num_mics, self.msm_setup.mics.num_mics)
        )

    def _get_eigmode_feature(self, f, num):
        return EigmodeFeature(
            freq_data = self.msm_setup.freq_data, f=f, num=num,
            shape = (self._get_fdim(f), self.msm_setup.mics.num_mics, self.msm_setup.mics.num_mics)
        )

    def _get_sourcemap_feature(self, f, num):
        return SourcemapFeature(
            beamformer = self.msm_setup.beamformer, f=f, num=num,
            shape = (self._get_fdim(f),) + self.msm_setup.grid.shape
        )

    def _get_source_strength_analytic_feature(self, f, num):
        return CSMDiagonalWelch(name="source_strength_analytic",
                    transfer=TransferMonopole(env=self.msm_setup.env, ref=self.msm_setup.ref),
                    freq_data=self.msm_setup._fft_ref_spectra, shape=(self._get_fdim(f), None),
                    mode="analytic", csm_type="source", f=f, num=num)

    def _get_source_strength_estimated_feature(self, f, num):
        return CSMDiagonalWelch(name="source_strength_estimated",
                    freq_data=self.msm_setup._fft_ref_spectra, shape=(self._get_fdim(f), None),
                    mode="welch", csm_type="source", f=f, num=num)

    def _get_noise_strength_analytic_feature(self, f, num):
        return CSMDiagonalWelch(name="noise_strength_analytic",
                    freq_data=self.msm_setup.freq_data, shape=(self._get_fdim(f), self.msm_setup.mics.num_mics),
                    mode="analytic", csm_type="noise", f=f, num=num)

    def _get_noise_strength_estimated_feature(self, f, num):
        return CSMDiagonalWelch(name="noise_strength_estimated",
                    freq_data=self.msm_setup.freq_data, shape=(self._get_fdim(f), self.msm_setup.mics.num_mics),
                    mode="welch", csm_type="noise", f=f, num=num)

    def _get_loc_feature(self):
        loc_feature = FloatFeature(name="loc", shape=(3, None))
        loc_feature.feature_func = lambda sampler, dtype, name: {
            name: sampler[4].target.astype(dtype)}
        return loc_feature

    def _get_targetmap_analytic_feature(self, loc_callable, strength_callable, f, num):
        return TargetmapFeature(
            name = "targetmap_analytic",
            shape = (self._get_fdim(f),) + self.msm_setup.grid.shape,
            grid = self.msm_setup.grid,
            loc_callable = loc_callable,
            strength_callable = strength_callable
        )

    def _get_targetmap_estimated_feature(self, loc_callable, strength_callable, f, num):
        return TargetmapFeature(
            name = "targetmap_estimated",
            shape = (self._get_fdim(f),) + self.msm_setup.grid.shape,
            grid = self.msm_setup.grid,
            loc_callable = loc_callable,
            strength_callable = strength_callable
        )

    def _get_f_feature(self, f, num):
        f_array = f_as_array(f, num, self.msm_setup.freq_data.fftfreq())
        feat = FloatFeature(name="f", shape=f_array.shape)
        feat.feature_func = lambda sampler, dtype, name: {name: f_array.astype(dtype)}
        return feat

    def _get_num_feature(self, num):
        feat = IntFeature(name="num", shape=(1,))
        feat.feature_func = lambda sampler, dtype, name: {name: np.array([num], dtype=dtype)}
        return feat

    def _get_fdim(self, f):
        if f is None:
            return self.msm_setup.freq_data.fftfreq().shape[0]
        elif isinstance(f, list):
            return len(f)
        else:
            return 1


class DatasetSyntheticConfigAnalytic(DatasetSyntheticConfig):
    """
    Default Configuration class for synthetic data simulation.

    Attributes
    ----------
    msm_setup : SyntheticSetupAnalytic
        SyntheticSetupAnalytic object.
    sampler_setup : SyntheticSamplerSetup
        SyntheticSamplerSetup object.
    """

    msm_setup = Instance(SyntheticSetupAnalytic, args=(), desc="SyntheticSetupAnalytic object.")

    def _get_default_msm_setup(self, **kwargs):
        kwargs["signal_length"] = kwargs.get("signal_length", 5.0)
        kwargs["source_grid"] = kwargs.get("source_grid", None)
        kwargs = self.get_default_mics(**kwargs)
        kwargs = self.get_default_env(**kwargs)
        kwargs = self.get_default_grid(**kwargs)
        kwargs = self.get_default_steer(**kwargs)
        kwargs = self.get_default_beamformer(**kwargs)
        kwargs = self.get_default_freq_data(**kwargs)

        setup_class_attr = SyntheticSetupAnalytic.class_traits().keys()
        msm_setup = SyntheticSetupAnalytic(
            **{key: kwargs.pop(key) for key in setup_class_attr if key in kwargs}
        )
        return msm_setup

    def get_default_freq_data(self, **kwargs):
        if kwargs.get("freq_data") is None:
            fs = kwargs.get("fs", 13720)
            signal_length = kwargs.get("signal_length", 5.0)
            transfer = TransferMonopole(
                env = kwargs["env"], ref = kwargs["beamformer"].steer.ref, mics=kwargs["mics"])
            kwargs["freq_data"] = PowerSpectraAnalytic(transfer = transfer,
                precision="complex64", sample_freq=13720, block_size=128, overlap="50%", mode=kwargs["mode"],
                numsamples=int(fs*signal_length))
        return kwargs

    def _apply_new_mic_pos(self,sampler, msm_setup):
        mic_sampler = sampler.get(1)
        if mic_sampler is not None:
            msm_setup.freq_data.transfer.mics = mic_sampler.target
        else:
            msm_setup.freq_data.transfer.mics = msm_setup.mics

    def _apply_new_seeds(self,sampler, msm_setup):
        seed_sampler = sampler.get(2)
        if seed_sampler is not None:
            msm_setup.freq_data.seed = seed_sampler.target

    def _apply_new_loc(self,sampler, msm_setup):
        loc = sampler[4].target
        msm_setup.freq_data.transfer.grid = ac.ImportGrid(gpos_file=loc)

    def _apply_new_signal_length(self,sampler, msm_setup):
        signal_length_sampler = sampler.get(6)
        if signal_length_sampler is not None:
            msm_setup.freq_data.numsamples = signal_length_sampler.target*msm_setup.freq_data.sample_freq

    def _apply_new_source_strength(self,sampler, msm_setup):
        nfft = msm_setup.freq_data.fftfreq().shape[0]
        nsources = sampler[4].nsources
        prms_sq_per_freq = sampler[3].target[:nsources]**2 / nfft
        msm_setup.freq_data.Q = np.stack([np.diag(prms_sq_per_freq) for _ in range(nfft)], axis=0)

    def _apply_new_mic_sig_noise(self,sampler, msm_setup):
        noise_sampler = sampler.get(5)
        if noise_sampler is not None:
            nfft = msm_setup.freq_data.fftfreq().shape[0]
            nsources = sampler[4].nsources
            prms_sq = sampler[3].target[:nsources]**2
            noise_signal_ratio = noise_sampler.target # normalized noise variance
            noise_prms_sq = prms_sq.sum()*noise_signal_ratio
            noise_prms_sq_per_freq = noise_prms_sq / nfft
            nperf = np.diag(np.array([noise_prms_sq_per_freq]*msm_setup.beamformer.steer.mics.num_mics))
            msm_setup.freq_data.noise = np.stack([nperf for _ in range(nfft)], axis=0)
        else:
            msm_setup.freq_data.noise = None

    def get_prepare_func(self):
        def prepare(sampler, msm_setup):
            if self.sampler_setup.mic_pos_noise:
                self._apply_new_mic_pos(sampler, msm_setup)
            self._apply_new_loc(sampler, msm_setup)
            self._apply_new_seeds(sampler, msm_setup)
            if self.sampler_setup.random_signal_length:
                self._apply_new_signal_length(sampler, msm_setup)
            self._apply_new_source_strength(sampler, msm_setup)
            if self.sampler_setup.mic_sig_noise:
                self._apply_new_mic_sig_noise(sampler, msm_setup)
            return {}
        return partial(prepare, msm_setup=self.msm_setup)

    def _get_source_strength_analytic_feature(self, f, num):
        return CSMDiagonalAnalytic(name="source_strength_analytic",
                    freq_data = self.msm_setup.freq_data,
                    shape=(self._get_fdim(f), None),
                    mode="analytic", csm_type="source", f=f, num=num)

    def _get_source_strength_estimated_feature(self, f, num):
        return CSMDiagonalAnalytic(name="source_strength_estimated",
                    freq_data = self.msm_setup.freq_data,
                    shape=(self._get_fdim(f), None),
                    mode="wishart", csm_type="source", f=f, num=num)

    def _get_noise_strength_estimated_feature(self, f, num):
        return CSMDiagonalAnalytic(name="noise_strength_estimated",
                    freq_data = self.msm_setup.freq_data,
                    shape=(self._get_fdim(f), self.msm_setup.mics.num_mics),
                    mode="wishart", csm_type="noise", f=f, num=num)

    def _get_noise_strength_analytic_feature(self, f, num):
        return CSMDiagonalAnalytic(name="noise_strength_analytic",
                    freq_data = self.msm_setup.freq_data,
                    shape=(self._get_fdim(f), self.msm_setup.mics.num_mics),
                    mode="analytic", csm_type="noise", f=f, num=num)

    def _get_time_data_feature(self, **kwargs):
        raise ValueError("time_data feature is not possible with modes ['analytic', 'wishart'].")

    def _get_spectrogram_feature(self, **kwargs):
        raise ValueError("spectrogram feature is not possible with modes ['analytic', 'wishart'].")

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        # create a source steer object for the source strength calculation
        self.msm_setup.build()

        builder = FeatureCollectionBuilder()
        # add prepare function
        builder.add_custom(self.get_prepare_func())
        builder.add_seeds(self.sampler_setup.numsampler)
        builder.add_idx()
        # add feature functions
        loc_feature = self._get_loc_feature()
        source_strength_analytic_feature = self._get_source_strength_analytic_feature(f, num)
        source_strength_estimated_feature = self._get_source_strength_estimated_feature(f, num)

        if "time_data" in features:
            builder.add_feature(self._get_time_data_feature())
        if "spectrogram" in features:
            builder.add_feature(self._get_spectrogram_feature(f, num))
        if "csm" in features:
            builder.add_feature(self._get_csm_feature(f, num))
        if "csmtriu" in features:
            builder.add_feature(self._get_csmtriu_feature(f, num))
        if "eigmode" in features:
            builder.add_feature(self._get_eigmode_feature(f, num))
        if "sourcemap" in features:
            builder.add_feature(self._get_sourcemap_feature(f, num))
        if "source_strength_analytic" in features:
            builder.add_feature(source_strength_analytic_feature)
        if "source_strength_estimated" in features:
            builder.add_feature(source_strength_estimated_feature)
        if "noise_strength_analytic" in features:
            builder.add_feature(self._get_noise_strength_analytic_feature(f, num))
        if "noise_strength_estimated" in features:
            builder.add_feature(self._get_noise_strength_estimated_feature(f, num))
        if "targetmap_analytic" in features:
            builder.add_feature(self._get_targetmap_analytic_feature(
                loc_feature.feature_func, source_strength_analytic_feature.feature_func, f, num))
        if "targetmap_estimated" in features:
            builder.add_feature(self._get_targetmap_estimated_feature(
                loc_feature.feature_func, source_strength_estimated_feature.feature_func, f, num))
        if "loc" in features:
            builder.add_feature(loc_feature)
        if "f" in features:
            builder.add_feature(self._get_f_feature(f, num))
        if "num" in features:
            builder.add_feature(self._get_num_feature(num))
        for feature in features:
            if isinstance(feature, Feature):
                self._feature_builder.add_feature(feature)
            elif not isinstance(feature, str):
                raise ValueError(f"Feature {feature} is not a valid feature object.")

        # if "source_csm_analytic" in features:
        #     builder.add_feature(
        #         CSMAnalytic(
        #             name="source_csm_analytic",
        #             freq_data = self.msm_setup.freq_data,
        #             shape = (fdim, mdim, mdim),
        #             mode = "analytic",
        #             csm_type = "source",
        #             f=f, num=num
        #         ))

        # if "source_csm_estimated" in features:
        #     builder.add_feature(
        #         CSMAnalytic(
        #             name="source_csm_estimated",
        #             freq_data = self.msm_setup.freq_data,
        #             shape = (fdim, mdim, mdim),
        #             mode = "wishart",
        #             csm_type = "source",
        #             f=f, num=num
        #         ))

        # if "noise_csm_analytic" in features:
        #     builder.add_feature(
        #         CSMAnalytic(
        #             name="noise_csm_analytic",
        #             freq_data = self.msm_setup.freq_data,
        #             shape = (fdim, mdim, mdim),
        #             mode = "analytic",
        #             csm_type = "noise",
        #             f=f, num=num
        #         ))


        # if "noise_csm_estimated" in features:
        #     builder.add_feature(
        #         CSMAnalytic(
        #             name="noise_csm_estimated",
        #             freq_data = self.msm_setup.freq_data,
        #             shape = (fdim, mdim, mdim),
        #             mode = "wishart",
        #             csm_type = "noise",
        #             f=f, num=num
        #         ))
        return builder.feature_collection


class DatasetSyntheticConfigWelch(DatasetSyntheticConfig):
    """
    Default Configuration class for synthetic data simulation.

    Attributes
    ----------
    msm_setup : SyntheticSetupWelch
        SyntheticSetupWelch object.
    sampler_setup : SyntheticSamplerSetup
        SyntheticSamplerSetup object.
    mode : Enum
        type of PowerSpectra calculation method.
    """

    msm_setup = Instance(SyntheticSetupWelch, desc="SyntheticSetupWelch object.")
    fs = DelegatesTo("msm_setup", prefix="_fs")

    def _get_default_msm_setup(self, **kwargs):
        kwargs["signal_length"] = kwargs.get("signal_length", 5.0)
        kwargs["source_grid"] = kwargs.get("source_grid", None)
        kwargs = self.get_default_mics(**kwargs)
        kwargs = self.get_default_env(**kwargs)
        kwargs = self.get_default_grid(**kwargs)
        kwargs = self.get_default_sources(**kwargs)
        kwargs = self.get_default_noise(**kwargs)
        kwargs = self.get_default_steer(**kwargs)
        kwargs = self.get_default_beamformer(**kwargs)
        kwargs = self.get_default_freq_data(**kwargs)

        setup_class_attr = SyntheticSetupWelch.class_traits().keys()
        msm_setup = SyntheticSetupWelch(
            **{key: kwargs.pop(key) for key in setup_class_attr if key in kwargs}
        )
        return msm_setup

    def get_default_noise(self, **kwargs):
        if kwargs.get("noise") is None:
            kwargs["noise"] = ac.UncorrelatedNoiseSource(
                signal = ac.WNoiseGenerator(
                    seed = 1000,
                    sample_freq = kwargs.get("fs", 13720),
                    numsamples = int(kwargs.get("fs", 13720)*kwargs.get("signal_length", 5.0)),
                    ),
                mics = kwargs["mics"]
                )
        return kwargs

    def get_default_sources(self, **kwargs):
        if kwargs.get("sources") is None:
            kwargs["sources"] = []
            for i in range(kwargs.get("max_nsources", 10)):
                kwargs["sources"].append(
                    ac.PointSource(signal = ac.WNoiseGenerator(
                            seed = i+1,
                            sample_freq = kwargs.get("fs", 13720),
                            numsamples = int(kwargs.get("fs", 13720)*kwargs.get("signal_length", 5.0)),
                            ),
                            env = kwargs["env"],
                            mics = kwargs["mics"],
                            )
                    )
        return kwargs

    def get_default_freq_data(self, **kwargs):
        if kwargs.get("freq_data") is None:
            fs = kwargs.get("fs", 13720)
            signal_length = kwargs.get("signal_length", 5.0)
            kwargs["freq_data"] = ac.PowerSpectra(
                source = ac.TimeSamples(sample_freq=fs, numsamples=int(fs*signal_length)), # dummy source
                precision="complex64", block_size=128, overlap="50%", window="Hanning", cached=False
            )
        return kwargs

    def _connect_objects(self, sampler, msm_setup):
        nsources = sampler[4].nsources
        if self.sampler_setup.mic_sig_noise:
            self.msm_setup.time_data.sources = msm_setup.sources[:nsources] + [msm_setup.noise]
        else:
            self.msm_setup.time_data.sources = msm_setup.sources[:nsources]

    def _apply_new_mic_pos(self, sampler, msm_setup):
        noisy_mics = sampler[1].target
        self.msm_setup.source_steer.mics = noisy_mics
        for src in msm_setup.sources:
            src.mics = noisy_mics

    def _apply_new_ref_mic(self, sampler, msm_setup):
        obs_sources = [deepcopy(src) for src in self.msm_setup.time_data.sources]
        ref = ac.MicGeom(mpos_tot=self.msm_setup.source_steer.ref[:,np.newaxis])
        for src in obs_sources:
            src.mics = ref
        self.msm_setup._fft_ref_spectra.source = ac.SourceMixer(sources=obs_sources)

    def _apply_new_loc(self, sampler, msm_setup):
        locations = sampler[4].target
        for i, loc in enumerate(locations.T):
            msm_setup.time_data.sources[i].loc = (loc[0], loc[1], loc[2])
        msm_setup.source_steer.grid = ac.ImportGrid(gpos_file=locations)

    def _apply_new_seeds(self, sampler, msm_setup):
        seed = sampler.get(2).target
        for i in range(sampler[4].nsources):
            msm_setup.time_data.sources[i].signal.seed = seed + i
        if self.sampler_setup.mic_sig_noise:
            msm_setup.time_data.sources[-1].signal.seed = seed + 1000

    def _apply_new_signal_length(self, sampler, msm_setup):
        signal_length_sampler = sampler.get(6)
        t = signal_length_sampler.target
        msm_setup.signal_length = t

    def _apply_new_source_strength(self, sampler, msm_setup):
        nsources = sampler[4].nsources # loc sampler
        rms = sampler[3].target[:nsources]*self.msm_setup.source_steer.r0
        for i in range(nsources):
            msm_setup.time_data.sources[i].signal.rms = rms[i] #TODO: use strength of a monopole source at ref! Not the transfer function itself

    def _apply_new_mic_sig_noise(self, sampler, msm_setup):
        noise_sampler = sampler.get(5)
        if noise_sampler is not None:
            nsources = sampler[4].nsources # loc sampler
            prms_sq = sampler[3].target[:nsources]**2
            noise_signal_ratio = noise_sampler.target # normalized noise variance
            noise_prms_sq = prms_sq.sum()*noise_signal_ratio
            msm_setup.time_data.sources[-1].signal.rms = np.sqrt(noise_prms_sq)

    def get_prepare_func(self):
        def prepare(sampler, msm_setup):
            self._connect_objects(sampler, msm_setup)
            if self.sampler_setup.mic_pos_noise:
                self._apply_new_mic_pos(sampler, msm_setup)
            self._apply_new_loc(sampler, msm_setup)
            self._apply_new_seeds(sampler, msm_setup)
            if self.sampler_setup.random_signal_length:
                self._apply_new_signal_length(sampler, msm_setup)
            self._apply_new_source_strength(sampler, msm_setup)
            if self.sampler_setup.mic_sig_noise:
                self._apply_new_mic_sig_noise(sampler, msm_setup)
            self._apply_new_ref_mic(sampler, msm_setup)
            return {}
        return partial(prepare, msm_setup=self.msm_setup)

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        # create additional objects needed for feature calculation and sampling
        self.msm_setup.source_steer = deepcopy(self.msm_setup.steer)
        if self.sampler_setup.mic_pos_noise:
            self.msm_setup.source_steer.mics = self.sampler_setup.mic_pos_sampler.target
        else:
            self.msm_setup.source_steer = self.msm_setup.steer
        for src in self.msm_setup.sources:
            src.mics = self.msm_setup.source_steer.mics
        self.msm_setup._fft_ref_spectra = ac.PowerSpectra(
            cached = False,
            window = self.msm_setup.freq_data.window,
            block_size = self.msm_setup.freq_data.block_size,
            overlap = self.msm_setup.freq_data.overlap,
            source = self.msm_setup.time_data
        )

        builder = FeatureCollectionBuilder()
        # add prepare function
        builder.add_custom(self.get_prepare_func())
        builder.add_seeds(self.sampler_setup.numsampler)
        builder.add_idx()
        # add feature functions
        loc_feature = self._get_loc_feature()
        source_strength_analytic_feature = self._get_source_strength_analytic_feature(f, num)
        source_strength_estimated_feature = self._get_source_strength_estimated_feature(f, num)

        if "time_data" in features:
            builder.add_feature(self._get_time_data_feature())
        if "spectrogram" in features:
            builder.add_feature(self._get_spectrogram_feature(f, num))
        if "csm" in features:
            builder.add_feature(self._get_csm_feature(f, num))
        if "csmtriu" in features:
            builder.add_feature(self._get_csmtriu_feature(f, num))
        if "eigmode" in features:
            builder.add_feature(self._get_eigmode_feature(f, num))
        if "sourcemap" in features:
            builder.add_feature(self._get_sourcemap_feature(f, num))
        if "source_strength_analytic" in features:
            builder.add_feature(source_strength_analytic_feature)
        if "source_strength_estimated" in features:
            builder.add_feature(source_strength_estimated_feature)
        if "noise_strength_analytic" in features:
            builder.add_feature(self._get_noise_strength_analytic_feature(f, num))
        if "noise_strength_estimated" in features:
            builder.add_feature(self._get_noise_strength_estimated_feature(f, num))
        if "targetmap_analytic" in features:
            builder.add_feature(self._get_targetmap_analytic_feature(
                loc_feature.feature_func, source_strength_analytic_feature.feature_func, f, num))
        if "targetmap_estimated" in features:
            builder.add_feature(self._get_targetmap_estimated_feature(
                loc_feature.feature_func, source_strength_estimated_feature.feature_func, f, num))
        if "loc" in features:
            builder.add_feature(loc_feature)
        if "f" in features:
            builder.add_feature(self._get_f_feature(f, num))
        if "num" in features:
            builder.add_feature(self._get_num_feature(num))
        for feature in features:
            if isinstance(feature, Feature):
                self._feature_builder.add_feature(feature)
            elif not isinstance(feature, str):
                raise ValueError(f"Feature {feature} is not a valid feature object.")
        return builder.feature_collection


def f_as_array(f, num, fftfreq):
    if f is None:
        f_array = fftfreq
    elif isinstance(f, list):
        if num == 0:
            f_array = np.array(
                [fftfreq[np.searchsorted(fftfreq, freq)] for freq in f])
        else:
            f_array = np.array(f)
    else:
        f_array = np.array([fftfreq[np.searchsorted(fftfreq, f)]])
    return f_array




# class DatasetSyntheticReverbConfig(DatasetSyntheticConfig):

#     # important note for the doc string:
#     # rms value belongs to the source signal and not to the measured signal at the reference position

#     def create_room_sampler(self):
#         aperture = self.mics.aperture
#         random_func = partial(sample_shoebox_room, aperture=aperture)
#         return sp.ContainerSampler(
#             random_func = random_func)

#     def create_relative_location_sampler(self):
#         """Create a sampler for the relative location of the sources in the room.

#         The sampler is a :class:`acoupipe.sampler.ContainerSampler` with a random function
#         that samples the relative location of the microphone array and the sources in the room.

#         Returns
#         -------
#         sp.ContainerSampler
#             The sampler for the relative location of the sources in the room.

#         """
#         def random_func(rng):
#             # Randomly sample the relative placement of the array and the sources inside the room
#             shift = np.array(
#                 [rng.uniform(0.05,0.95), rng.uniform(0.05,0.95), rng.uniform(0.05,0.95)]
#             )
#             return shift
#         return sp.ContainerSampler(random_func = random_func)

#     def get_sampler(self):
#         location_sampler = self.create_location_sampler()
#         sampler = {
#             2 : self.create_signal_seed_sampler(),
#             3 : self.create_rms_sampler(),
#             4 : location_sampler,
#             7 : self.create_room_sampler(),
#             8 : self.create_relative_location_sampler(),
#             }
#         if self.max_nsources != self.min_nsources:
#             sampler[0] = self.create_nsources_sampler(
#                 target=[location_sampler])
#         if self.mic_pos_noise:
#             sampler[1] = self.create_micgeom_sampler()
#         if self.mic_sig_noise:
#             sampler[5] = self.create_mic_noise_sampler()
#         if self.random_signal_length:
#             sampler[6] = self.create_signal_length_sampler()
#         return sampler

#     def create_sources(self):
#         sources = []
#         for signal in self.signals:
#             sources.append(
#                 ac.PointSourceConvolve(
#                     signal=signal,
#                     mics=self._noisy_mics,
#                     env=self.env,)
#             )
#         return sources

#     @staticmethod
#     def _get_rir_matrix(room, locs, mics, pad):
#         """Compute the RIRs for the given room, source and microphone locations.

#         Parameters
#         ----------
#         room : pyroomacoustics.room.ShoeBox
#             The room object.
#         locs : np.ndarray
#             The source locations with shape (3, n_sources).
#         mics : np.ndarray
#             The microphone locations with shape (3, n_mics).
#         pad : bool
#             If True, pad the RIRs to the length of the longest RIR. If False, trim the RIRs to the
#             length of the shortest RIR.
#         """
#         room.compute_rir() # room.rir is a list of lists of RIRs with different length
#         n = len(room.rir[0][0]) # length of the first rir
#         if not pad:
#             # trim all RIRs by the length of the shortest one
#             rir_arr = np.zeros((locs.shape[1], mics.shape[1],n))
#             for j in range(locs.shape[1]):
#                 for i in range(mics.shape[1]):
#                     rir = np.array(room.rir[i][j])
#                     ns = min(n, rir.shape[0])
#                     rir_arr[j,i,:ns] = rir[:ns]
#             return rir_arr[:,:,:ns]
#         else:
#             for j in range(locs.shape[1]):
#                 for i in range(mics.shape[1]):
#                     n = max(n, np.array(room.rir[i][j]).shape[0])
#             rir_arr = np.zeros((locs.shape[1], mics.shape[1],n))
#             for j in range(locs.shape[1]):
#                 for i in range(mics.shape[1]):
#                     rir = np.array(room.rir[i][j])
#                     ns = rir.shape[0]
#                     rir_arr[j,i,:ns] = rir
#         return rir_arr

#     @staticmethod
#     def _get_loc_shift(locs, mics, walls, rloc):
#         pmax = np.max(np.concatenate([locs, mics], axis=1),axis=1)
#         pmin = np.min(np.concatenate([locs, mics], axis=1),axis=1)
#         rmax = np.array([w.corners.max(1) for w in walls]).max(0)
#         return rloc*(rmax - pmax - pmin)

#     @staticmethod
#     def _get_rir(sampler, beamformer):
#         micgeom_sampler = sampler.get(1)
#         room_sampler = sampler.get(7)
#         loc_sampler = sampler.get(4)
#         relative_location_sampler = sampler.get(8)
#         freq_data = beamformer.freq_data

#         # set up microphone array
#         if micgeom_sampler is not None:
#             noisy_mpos = micgeom_sampler.target.mpos
#         else:
#             noisy_mpos = beamformer.steer.mics.mpos # use the original mics (without noise)

#         # creating the room
#         room = room_sampler.target
#         room.fs = int(freq_data.sample_freq)

#         # oriantation of the microphone array and sources in the room
#         loc = loc_sampler.target
#         all_mics = np.concatenate([noisy_mpos, beamformer.steer.ref[:,np.newaxis]], axis=1)
#         shift = DatasetSyntheticReverbConfig._get_loc_shift(loc, all_mics, room.walls, relative_location_sampler.target)

#         # set the sources and microphones
#         room.sources = [pra.SoundSource(loc+shift) for loc in loc.T]
#         room.add_microphone_array(
#             pra.beamforming.MicrophoneArray(all_mics+shift[:,np.newaxis], freq_data.sample_freq)
#         )
#         # get the RIR
#         rir = DatasetSyntheticReverbConfig._get_rir_matrix(room, loc, all_mics, pad=True)

#         # trim the RIRs to the a power of 2 length
#         lpow2 = 2**int(np.ceil(np.log2(rir.shape[2])))
#         padded_rir = np.zeros((rir.shape[0], rir.shape[1], lpow2))
#         padded_rir[:,:,:rir.shape[2]] = rir
#         return padded_rir


#     @staticmethod
#     def calc_welch_prepare_func(sampler, beamformer, sources, source_steer, fft_spectra, fft_ref_spectra):
#         # restore sampler and acoular objects
#         micgeom_sampler = sampler.get(1)
#         seed_sampler = sampler.get(2)
#         rms_sampler = sampler.get(3)
#         loc_sampler = sampler.get(4)
#         noise_sampler = sampler.get(5)
#         signal_length_sampler = sampler.get(6)

#         freq_data = beamformer.freq_data

#         if micgeom_sampler is not None:
#             noisy_mics = micgeom_sampler.target
#         else:
#             noisy_mics = beamformer.steer.mics # use the original mics (without noise)

#         if signal_length_sampler is not None:
#             # adjust source signals, noise signal length
#             signals = get_all_source_signals(sources)
#             for signal in signals:
#                 signal.numsamples = signal_length_sampler.target*freq_data.sample_freq
#         # sample parameters
#         loc = loc_sampler.target
#         nsources = loc.shape[1]
#         prms_sq = rms_sampler.target[:nsources]**2 # squared sound pressure RMS at reference position
#         # apply parameters
#         mic_noise = get_uncorrelated_noise_source_recursively(freq_data.source)
#         if mic_noise:
#             mic_noise_signal = mic_noise[0].signal
#             if signal_length_sampler is not None:
#                 mic_noise_signal.numsamples = signal_length_sampler.target*freq_data.sample_freq
#             if noise_sampler is not None:
#                 noise_signal_ratio = noise_sampler.target # normalized noise variance
#                 noise_prms_sq = prms_sq.sum()*noise_signal_ratio
#                 mic_noise_signal.rms = np.sqrt(noise_prms_sq)
#                 mic_noise_signal.seed = seed_sampler.target+1000
#                 freq_data.source.source.mics = noisy_mics
#         subset_sources = sources[:nsources]
#         # creating the room
#         rir = DatasetSyntheticReverbConfig._get_rir(sampler, beamformer)
#         for i,src in enumerate(subset_sources):

#             #src.kernel = rir[i,:-1].T
#             # NOTE: this step leads to numerically unstable results
#             # it may be better to directly use the transfer function of the room
#             # without normalization. However, this requires fundamental changes with the
#             # label calculation
#             tf = blockwise_transfer(rir[i]).T
#             #tf /= tf[:,-1][:,np.newaxis] # reference mic based normalization
#             # ifft to get kernel
#             src.kernel = np.fft.irfft(tf[:,:-1],axis=0)
#             src.signal.seed = seed_sampler.target+i
#             src.signal.rms = np.sqrt(prms_sq[i])
#             src.loc = (loc[0,i],loc[1,i],loc[2,i] )
#         freq_data.source.sources = subset_sources # apply subset of sources
#         fft_spectra.source = freq_data.source # only for spectrogram feature
#         # update observation point
#         obs_sources = deepcopy(subset_sources)
#         kernel = np.zeros((freq_data.block_size, 1))
#         kernel[0] = 1
#         ref = ac.MicGeom(mpos_tot=source_steer.ref[:,np.newaxis])
#         for i, src in enumerate(obs_sources):
#             src.mics = ref
#             #src.kernel = np.fft.irfft(tf[:,-1][:,np.newaxis],axis=0)
#             src.kernel = kernel
#             # src.kernel = rir[i,-1][np.newaxis].T
#         fft_ref_spectra.source = ac.SourceMixer(sources=obs_sources)
#         return {}

#     @staticmethod
#     def calc_analytic_prepare_func(sampler, beamformer):
#         micgeom_sampler = sampler.get(1)
#         seed_sampler = sampler.get(2)
#         rms_sampler = sampler.get(3)
#         loc_sampler = sampler.get(4)
#         noise_sampler = sampler.get(5)
#         signal_length_sampler = sampler.get(6)

#         freq_data = beamformer.freq_data

#         if micgeom_sampler is not None:
#             noisy_mics = micgeom_sampler.target
#         else:
#             noisy_mics = beamformer.steer.mics # use the original mics (without noise)

#         if signal_length_sampler is not None:
#             freq_data.numsamples = signal_length_sampler.target*freq_data.sample_freq

#         nfft = freq_data.fftfreq().shape[0]
#         nummics = beamformer.steer.mics.num_mics
#         # sample parameters
#         loc = loc_sampler.target
#         nsources = loc.shape[1]
#         freq_data.transfer.grid = ac.ImportGrid(gpos_file=loc) # set source locations
#         freq_data.transfer.mics = noisy_mics # set mic locations
#         freq_data.seed=seed_sampler.target
#         # change source strength
#         prms_sq = rms_sampler.target[:nsources]**2 # squared sound pressure RMS at reference position
#         prms_sq_per_freq = prms_sq / nfft #prms_sq_per_freq
#         freq_data.Q = np.stack([np.diag(prms_sq_per_freq) for _ in range(nfft)], axis=0)
#         # add noise to freq_data
#         if noise_sampler is not None:
#             noise_signal_ratio = noise_sampler.target # normalized noise variance
#             noise_prms_sq = prms_sq.sum()*noise_signal_ratio
#             noise_prms_sq_per_freq = noise_prms_sq / nfft
#             nperf = np.diag(np.array([noise_prms_sq_per_freq]*nummics))
#             freq_data.noise = np.stack([nperf for _ in range(nfft)], axis=0)
#         else:
#             freq_data.noise = None
#         # calculate the transfer function
#         transfer = np.empty((nfft, nummics,nsources), dtype=complex)
#         rir = DatasetSyntheticReverbConfig._get_rir(sampler, beamformer)
#         for i in range(nsources):
#             tf = blockwise_transfer(rir[i], freq_data.block_size).T
#             #transfer[:,:,i] = tf[:,:-1] / tf[:,-1][:,np.newaxis] # reference mic based normalization
#             transfer[:,:,i] = tf[:,:-1]
#         freq_data.custom_transfer = transfer
#         return {}






# class DatasetSyntheticReverb(DatasetSynthetic):

#     def __init__(self, mode="welch", mic_pos_noise=True, mic_sig_noise=True,
#                 snap_to_grid=False, random_signal_length=False, signal_length=5, fs=13720., min_nsources=1,
#                 max_nsources=10, tasks=1, logger=None, config=None):
#         if config is None:
#             config = DatasetSyntheticReverbConfig(
#                 mode=mode, signal_length=signal_length, fs=fs,
#                 min_nsources=min_nsources, max_nsources=max_nsources,
#                 mic_pos_noise=mic_pos_noise, mic_sig_noise=mic_sig_noise,
#                 snap_to_grid=snap_to_grid, random_signal_length=random_signal_length)
#         super().__init__(tasks=tasks, logger=logger, config=config)








