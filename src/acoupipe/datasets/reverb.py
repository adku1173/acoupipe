"""Contains classes for the generation of microphone array data from synthesized signals for acoustic testing applications.

    Currently, the following dataset generators are available:

    * :class:`DatasetSynthetic`: A simple and fast method that relies on synthetic white noise signals and spatially stationary sources radiating under anechoic conditions.

.. _default measurement setup:

.. figure:: ../../../../_static/msm_layout.png
    :width: 750
    :align: center

    Default measurement setup used in the :py:mod:`acoupipe.datasets.synthetic` module.

"""


from functools import partial

import numpy as np
from traits.api import Enum, Instance

import acoupipe.sampler as sp
from acoupipe.datasets.setup import ISMSamplerSetup
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.synthetic import (
    DatasetSynthetic,
    DatasetSyntheticConfig,
    DatasetSyntheticConfigAnalytic,
    DatasetSyntheticConfigWelch,
)
from acoupipe.datasets.utils import get_absorption_coeff


class DatasetISM(DatasetSynthetic):
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
                config = DatasetISMConfigAnalytic(**kwargs)
            else:
                config = DatasetSyntheticConfigWelch(**kwargs)
        super().__init__(config=config, tasks=tasks, logger=logger)


class DatasetISMConfig(DatasetSyntheticConfig):

    sampler_setup = Instance(ISMSamplerSetup, desc="SyntheticSamplerSetup object.")
    ism_backend = Enum(("pyroomacoustics", "gpuRIR"), desc="Backend for the room simulation.")

    def _get_transfer_class(self):
        if self.ism_backend == "pyroomacoustics":
            from acoupipe.datasets.spectra_analytic import TransferShoeBoxPyroomacoustics
            return TransferShoeBoxPyroomacoustics
        elif self.ism_backend == "gpuRIR":
            from acoupipe.datasets.spectra_analytic import TransferShoeBoxGPUrir
            return TransferShoeBoxGPUrir

    def get_default_room_size_sampler(self, msm_setup, **kwargs):
        if kwargs.get("room_size_sampler") is None:
            ap = msm_setup.mics.aperture
            def random_func(rng):
                return np.array([
                round(rng.uniform(3*ap, 10*ap), 1),
                round(rng.uniform(3*ap, 10*ap), 1),
                round(rng.uniform(2*ap, 4.5*ap), 1),
                ])
            kwargs["room_size_sampler"] = sp.ContainerSampler(random_func = random_func)
        return kwargs

    def get_default_absoption_coeff_sampler(self, msm_setup, **kwargs):
        if kwargs.get("absoption_coeff_sampler") is None:
            random_func = partial(get_absorption_coeff, realistic_walls=True)
            kwargs["absoption_coeff_sampler"] = sp.ContainerSampler(
                random_func = random_func)
        return kwargs

    def get_default_room_placement_sampler(self, msm_setup, **kwargs):
        """Create a sampler for the relative location of the sources in the room.

        The sampler is a :class:`acoupipe.sampler.ContainerSampler` with a random function
        that samples the relative location of the microphone array and the sources in the room.

        Returns
        -------
        sp.ContainerSampler
            The sampler for the relative location of the sources in the room.

        """
        if kwargs.get("room_placement_sampler") is None:
            def random_func(rng):
                # Randomly sample the relative placement of the array and the sources inside the room
                shift = np.array(
                    [rng.uniform(0.05,0.95), rng.uniform(0.05,0.95), rng.uniform(0.05,0.95)]
                )
                return shift
            kwargs["room_placement_sampler"] = sp.ContainerSampler(random_func = random_func)
        return kwargs

    def get_default_freq_data(self, **kwargs):
        if kwargs.get("freq_data") is None:
            fs = kwargs.get("fs", 13720)
            signal_length = kwargs.get("signal_length", 5.0)
            Transfer = self._get_transfer_class()
            transfer = Transfer(env = kwargs["env"], ref = 1.0, mics=kwargs["mics"])
            kwargs["freq_data"] = PowerSpectraAnalytic(transfer = transfer,
                precision="complex64", sample_freq=13720, block_size=128, overlap="50%", mode=kwargs["mode"],
                numsamples=int(fs*signal_length))
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
        kwargs = self.get_default_room_size_sampler(msm_setup, **kwargs)
        kwargs = self.get_absoption_coeff_sampler(msm_setup, **kwargs)
        kwargs = self.get_default_room_placement_sampler(msm_setup, **kwargs)

        setup_class_attr = ISMSamplerSetup.class_traits().keys()
        sampler_setup = ISMSamplerSetup(msm_setup=msm_setup,
            **{key: kwargs.pop(key) for key in setup_class_attr if key in kwargs}
        )
        return sampler_setup



class DatasetISMConfigAnalytic(DatasetISMConfig, DatasetSyntheticConfigAnalytic):

    def _apply_room_settings(self, sampler, msm_setup):
        msm_setup.transfer.room_size = sampler[7]
        msm_setup.transfer.alpha = sampler[8]
        msm_setup.transfer.rel_tdir = sampler[9]

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
