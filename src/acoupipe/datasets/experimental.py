"""Contains classes for the generation of microphone array datasets with experimentally acquired signals for acoustic testing applications.

    Currently, the following dataset generators are available:

    * :class:`DatasetMIRACLE`: A microphone array dataset generator, relying on measured spatial room impulse responses from the `MIRACLE`_ dataset and synthetic white noise signals.

.. _measurement setup:

.. figure:: ../../../../_static/msm_miracle.png
    :width: 750
    :align: center

    Measurement setup `R2` from the `MIRACLE`_ dataset.

"""


from acoupipe.datasets.base import DatasetBase
from acoupipe.datasets.experimental_old import DatasetMIRACLEConfig

link_address = {
"A1" : "https://depositonce.tu-berlin.de/bitstreams/67156d9c-224d-4d07-b923-be0240e7b48d/download",
"A2" : "https://depositonce.tu-berlin.de/bitstreams/cbb462d7-cb28-4803-98d8-84b03aad0d5f/download",
"D1" : "https://depositonce.tu-berlin.de/bitstreams/86680ee5-ae0c-4b38-8ef8-805652a21ded/download",
"R2" : "https://depositonce.tu-berlin.de/bitstreams/0fc5f5a4-a2f7-4eb7-b796-7114260e5e86/download",
}

file_hash = {
"A1" : "b0e053319fabad6964e2275f4bcd2dcfc6f0dc5f463e0324b7ad107e76612f88",
"A2" : "c021cc57bb51237283c5303e235495edfea75b1f0eaba4a8f988942b9913e7ff",
"D1" : "d888201065a43f436080da470f025c245b1a8030e08ea7a9dce1dc6b160761ee",
"R2" : "479af6bfdd403c855d53304b291c4878c1f8d4a4482836de77677c03ffb6bbaa",
}

class DatasetMIRACLE(DatasetBase):
    r"""A microphone array dataset generator using experimentally measured data.

    DatasetSynthetic relies on measured spatial room impulse responses (SRIRs) from the `MIRACLE`_ dataset.

    MIRACLE is a SRIR dataset explicitly designed for acoustic testing applications using a planar microphone array focused on a
    rectangular observation area. It consists of a total of 856, 128 captured spatial room impulse responses and dense spatial sampling of
    the observation area.

    The data generation process is similar to :class:`acoupipe.datasets.synthetic.DatasetSynthetic`, but uses measured
    transfer functions / impulse responses instead of analytic ones. Multi-source scenarios with possibly closing neighboring sources are
    realized by superimposing signals that have been convolved with the provided SRIRs.

    **Scenarios**

    The MIRACLE dataset provides SRIRs from different measurement setups with the same microphone array,
    which can be selected by the :code:`scenario` parameter.
    The underlying measurement setup for :code:`scenario="R2"` is shown in the `measurement setup`_ figure.

    .. list-table:: Available scenarios
        :header-rows: 1
        :widths: 5 10 10 10 10 10 10

        *   - Scenario
            - Download Size
            - Environment
            - c0
            - # SRIRs
            - Source-plane dist.
            - Spatial sampling
        *   - A1
            - 1.1 GB
            - Anechoic
            - 344.7 m/s
            - 4096
            - 73.4 cm
            - 23.3 mm
        *   - D1
            - 300 MB
            - Anechoic
            - 344.8 m/s
            - 4096
            - 73.4 cm
            - 5.0 mm
        *   - A2
            - 1.1 GB
            - Anechoic
            - 345.0 m/s
            - 4096
            - 146.7 cm
            - 23.3 mm
        *   - R2
            - 1.1 GB
            - Reflective Ground
            - 345.2 m/s
            - 4096
            - 146.7 cm
            - 23.3 mm


    **Default FFT parameters**

    The underlying default FFT parameters are:

    .. table:: FFT Parameters

        ===================== ========================================
        Sampling Rate         fs=32,000 Hz
        Block size            256 Samples
        Block overlap         50 %
        Windowing             von Hann / Hanning
        ===================== ========================================

    **Default randomized properties**

    Several properties of the dataset are randomized for each source case when generating the data. This includes the number of sources,
    their positions, and strength. Their respective distributions, are closely related to :cite:`Herold2017`.
    Uncorrelated white noise is added to the microphone channels by default. Note that the source positions are sampled from a grid
    according to the spatial sampling of the MIRACLE dataset.

    .. table:: Randomized properties

        ==================================================================   ===================================================
        No. of Sources                                                       Poisson distributed (:math:`\lambda=3`)
        Source Positions [m]                                                 Bivariate normal distributed (:math:`\sigma = 0.1688 d_a`)
        Source Strength (:math:`[{Pa}^2]` at reference position)               Rayleigh distributed (:math:`\sigma_{R}=5`)
        Relative Noise Variance                                              Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
        ==================================================================   ===================================================

    Example
    -------

    This is a quick example on how to use the :class:`acoupipe.datasets.experimental.DatasetMIRACLE` dataset for generation of source cases
    with multiple sources. First, import the class and instantiate. One can either specify the path, where the SRIR files from the MIRACLE_
    project are stored, or one can set `srir_dir=None`. The latter will download the corresponding SRIR dataset into a pre-defined cache directory determined
    by the `pooch` library.

    .. code-block:: python

        from acoupipe.datasets.experimental import DatasetMIRACLE

        srir_dir = None
        # srir_dir = <local path to the MIRACLE dataset>

        dataset = DatasetMIRACLE(scenario='A1', mode='wishart')

    Now, extract the :code:`sourcmap` feature iteratively with:

    .. code-block:: python

        dataset_generator = dataset.generate(size=10, f=2000, features=['sourcemap','loc','f'], split='training')

        data_sample = next(dataset_generator)

    And finally, plot the results:

    .. code-block:: python

        import acoular as ac
        import matplotlib.pyplot as plt
        import numpy as np

        extent = dataset.config.grid.extend()

        # sound pressure level
        Lm = ac.L_p(data_sample['sourcemap']).T
        Lm_max = Lm.max()
        Lm_min = Lm.max() - 20

        # plot sourcemap
        plt.figure()
        plt.title(f'Beamforming Map (f={data_sample["f"][0]} Hz, scenario={dataset.config.scenario})')
        plt.imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin='lower')
        plt.colorbar(label='Sound Pressure Level (dB)')
        # plot source locations
        for loc in data_sample['loc'].T:
            plt.scatter(loc[0], loc[1])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()

    The resulting plot for the different scenarios should look like this:

        .. figure:: ../../../../_static/exp_sourcemap_example.png
            :width: 750
            :align: center


    **Initialization Parameters**
    """

    def __init__(self, srir_dir=None, scenario="A1", ref_mic_index=63, mode="welch", mic_sig_noise=True,
                random_signal_length=False, signal_length=5, min_nsources=1, max_nsources=10,
                tasks=1, config=None):
        """Initialize the DatasetMIRACLE object.

        Input parameters are passed to the DatasetMIRACLEConfig object, which creates
        all necessary objects for the simulation of microphone array data.

        Parameters
        ----------
        srir_dir : str, optional
            Path to the directory where the SRIR files are stored. Default is None, which
            sets the path to the `pooch.os_cache` directory. The SRIR files are downloaded from the
            `MIRACLE`_ dataset if they are not found in the directory.
        scenario : str, optional
            Scenario of the dataset. Possible values are "A1", "D1", "A2", "R2".
        ref_mic_index : int, optional
            Index of the microphone that is used as reference observation point.
            Default is 63, which is the index of the centermost microphone.
        mode : str, optional
            Mode of the dataset. Possible values are "analytic", "welch", "wishart".
            Default is "welch".
        mic_sig_noise : bool, optional
            Add uncorrelated noise to the microphone signals. Default is True.
        signal_length : float, optional
            Length of the signal in seconds. Default is 5.
        min_nsources : int, optional
            Minimum number of sources per sample. Default is 1.
        max_nsources : int, optional
            Maximum number of sources per sample. Default is 10.
        tasks : int, optional
            Number of parallel processes. Default is 1.
        config : DatasetMIRACLEConfig, optional
            DatasetMIRACLEConfig object. Default is None, which creates a new DatasetMIRACLEConfig object.
        """
        if config is None:
            config = DatasetMIRACLEConfig(
                mode=mode, random_signal_length=random_signal_length, signal_length=signal_length,min_nsources=min_nsources,
                max_nsources=max_nsources,srir_dir=srir_dir, scenario=scenario, ref_mic_index=ref_mic_index, mic_sig_noise=mic_sig_noise)
        super().__init__(tasks=tasks, config=config)


# class DatasetMIRACLEConfig(ConfigBase):

#     msm_setup = Instance(SyntheticSetup, desc="SyntheticSetupAnalytic object.")
#     sampler_setup = Instance(SyntheticSamplerSetup, desc="SyntheticSamplerSetup object.")

#     def get_default_sampler_setup(self, msm_setup, **kwargs):
#         if kwargs.get("mic_pos_sampler") is None:
#             kwargs["mic_pos_sampler"] = sp.MicGeomSampler(
#                 random_var = norm(loc=0, scale=0.001),
#                 ddir = np.array([[1.0], [1.0], [0]]),
#                 target = deepcopy(msm_setup.mics),
#                 mpos_init = msm_setup.mics.mpos_tot,
#             )
#         if kwargs.get("loc_sampler") is None:
#             ap = msm_setup.mics.aperture
#             kwargs["loc_sampler"] = sp.LocationSampler(
#                 random_var = (norm(0,0.1688*ap),norm(0,0.1688*ap),norm(0.5*ap,0)),
#                 x_bounds = (-0.5*ap,0.5*ap),
#                 y_bounds = (-0.5*ap,0.5*ap),
#                 z_bounds = (0.5*ap,0.5*ap),
#                 )

#         if kwargs.get("nsources_sampler") is None:
#             kwargs["nsources_sampler"] = sp.NumericAttributeSampler(
#                 random_var = poisson(mu=3, loc=1),
#                 attribute = "nsources",
#                 equal_value = True,
#                 target = [kwargs["loc_sampler"]],
#                 )

#         if kwargs.get("strength_sampler") is None:
#             kwargs["strength_sampler"] = sp.ContainerSampler(
#                 random_func = lambda rng: None, # placeholder
#             )

#         if kwargs.get("mic_sig_noise_sampler") is None:
#             kwargs["mic_sig_noise_sampler"] = sp.ContainerSampler(
#                 random_func = lambda rng: rng.uniform(10e-6,0.1)
#             )

#         if kwargs.get("signal_length_sampler") is None:
#             kwargs["signal_length_sampler"] = sp.ContainerSampler(
#                 random_func = lambda rng: rng.uniform(1, 10), # between 1 and 10 seconds
#             )

#         if kwargs.get("seed_sampler") is None:
#             kwargs["seed_sampler"] = sp.ContainerSampler(
#                 random_func = lambda rng: int(rng.uniform(1,1e9))
#             )

#         sampler_setup = SyntheticSamplerSetup(
#             msm_setup=msm_setup,
#             mic_pos_sampler=kwargs.pop("mic_pos_sampler"),
#             loc_sampler=kwargs.pop("loc_sampler"),
#             nsources_sampler=kwargs.pop("nsources_sampler"),
#             strength_sampler=kwargs.pop("strength_sampler"),
#             mic_sig_noise_sampler=kwargs.pop("mic_sig_noise_sampler"),
#             signal_length_sampler=kwargs.pop("signal_length_sampler"),
#             seed_sampler=kwargs.pop("seed_sampler"),
#         )
#         sampler_setup.max_nsources = kwargs.pop("max_nsources", 10)
#         sampler_setup.min_nsources = kwargs.pop("min_nsources", 1)
#         sampler_setup.strength_random_func = kwargs.pop("strength_random_func", lambda rng: np.sqrt(
#                                     rng.rayleigh(5,sampler_setup.max_nsources))) # samples rms values

#         setup_class_attr = SyntheticSamplerSetup.class_traits().keys()
#         for key in setup_class_attr:
#             if key in kwargs:
#                 setattr(sampler_setup, key, kwargs[key])
#         return sampler_setup





