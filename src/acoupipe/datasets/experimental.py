"""Contains classes for the generation of microphone array datasets with experimentally acquired signals for acoustic testing applications.

    Currently, the following dataset generators are available:

    * :class:`DatasetMIRACLE`: A microphone array dataset generator, relying on measured spatial room impulse responses from the `MIRACLE`_ dataset and synthetic white noise signals.

.. _measurement setup:

.. figure:: ../../../../_static/msm_miracle.png
    :width: 750
    :align: center

    Measurement setup `R2` from the `MIRACLE`_ dataset.

"""

from functools import partial
from pathlib import Path

import acoular as ac
import h5py as h5
import numpy as np
import pooch
from traits.api import Dict, Either, Enum, Instance, Int, Property, Str, observe

from acoupipe.datasets.base import DatasetBase
from acoupipe.datasets.synthetic import DatasetSyntheticConfig
from acoupipe.datasets.utils import (
    calc_transfer,
)

link_address = {
    'A1': 'https://depositonce.tu-berlin.de/bitstreams/67156d9c-224d-4d07-b923-be0240e7b48d/download',
    'A2': 'https://depositonce.tu-berlin.de/bitstreams/cbb462d7-cb28-4803-98d8-84b03aad0d5f/download',
    'D1': 'https://depositonce.tu-berlin.de/bitstreams/86680ee5-ae0c-4b38-8ef8-805652a21ded/download',
    'R2': 'https://depositonce.tu-berlin.de/bitstreams/0fc5f5a4-a2f7-4eb7-b796-7114260e5e86/download',
    'SR1-C1': 'https://depositonce.tu-berlin.de/bitstreams/75b3da38-9f7f-4b30-8967-3da1b0f29e9f/download',
    'SR1-C2': 'https://depositonce.tu-berlin.de/bitstreams/8651dc9d-ec9c-4504-b301-7b5536fb74f3/download',
    'SR1-C3': 'https://depositonce.tu-berlin.de/bitstreams/7c840fc0-90d5-40c1-8528-356b808560c8/download',
    'SR1-C4': 'https://depositonce.tu-berlin.de/bitstreams/5bbf3409-205e-451b-abda-c16e90004f2a/download',
    'SR1-D': 'https://depositonce.tu-berlin.de/bitstreams/412b04ec-6852-417f-ab85-3e847133860e/download',
    'SR2-C1': 'https://depositonce.tu-berlin.de/bitstreams/bff1df6c-2672-4352-aa80-34bb2aa7dfd1/download',
    'SR2-C2': 'https://depositonce.tu-berlin.de/bitstreams/12e6d24a-5d55-415c-b461-fa10d850c63c/download',
    'SR2-C3': 'https://depositonce.tu-berlin.de/bitstreams/b64c3d35-c337-4003-859d-e747bdbe2a3f/download',
    'SR2-C4': 'https://depositonce.tu-berlin.de/bitstreams/272bd891-3463-4459-be03-9a66aa916841/download',
    'SR2-D': 'https://depositonce.tu-berlin.de/bitstreams/ec51a7e5-818d-463c-8f77-342c3bde27d8/download',
    'SRA1-C1': 'https://depositonce.tu-berlin.de/bitstreams/bfbda52e-4776-4499-a52f-92c01dd0f632/download',
    'SRA1-C2': 'https://depositonce.tu-berlin.de/bitstreams/f67da2d7-5c57-4ddc-8230-3f275563c497/download',
    'SRA1-C3': 'https://depositonce.tu-berlin.de/bitstreams/ff5cb621-402d-43a6-8ff8-bbbd8d8ae170/download',
    'SRA1-C4': 'https://depositonce.tu-berlin.de/bitstreams/24c0f12b-dbdd-4c23-8825-6657ed7249e4/download',
    'SRA1-D': 'https://depositonce.tu-berlin.de/bitstreams/7950399f-e1be-4067-b672-b1c2b255a33b/download',
    'SRA2-C1': 'https://depositonce.tu-berlin.de/bitstreams/aa9d52fd-e7eb-4f38-93d8-e22733c8311f/download',
    'SRA2-C2': 'https://depositonce.tu-berlin.de/bitstreams/acb8baf0-7442-4faf-895d-7862004ffadc/download',
    'SRA2-C3': 'https://depositonce.tu-berlin.de/bitstreams/7351f03d-243b-4c52-97da-b9b45cdce6d4/download',
    'SRA2-C4': 'https://depositonce.tu-berlin.de/bitstreams/8b33b41b-51ca-44f6-a6b9-dac887d0dc7e/download',
    'SRA2-D': 'https://depositonce.tu-berlin.de/bitstreams/61016fe8-96fe-463d-9441-d4c39b0a9898/download',
}

file_hash = {
    'A1': 'b0e053319fabad6964e2275f4bcd2dcfc6f0dc5f463e0324b7ad107e76612f88',
    'A2': 'c021cc57bb51237283c5303e235495edfea75b1f0eaba4a8f988942b9913e7ff',
    'D1': 'd888201065a43f436080da470f025c245b1a8030e08ea7a9dce1dc6b160761ee',
    'R2': '479af6bfdd403c855d53304b291c4878c1f8d4a4482836de77677c03ffb6bbaa',
    'SR1-C1': 'b969ae330c585fc2ba4f1ccf9b8fb5be7756cd2b25d5762149705791964d721a',
    'SR1-C2': 'c5db8fb0530bc723c84cb1cde8ab60efb3ba52b2d25fd44601b36dedbf5726cf',
    'SR1-C3': '4850dc71bce9862269718d9190197718699c0cdf75b3e12255c717e7bc2b2d28',
    'SR1-C4': '8b0b2734ff4ad7c0065c39df135905d21c6d15c5e019d2cdc91ccd6686c67cc2',
    'SR1-D': '59bdc474a910fe2099a003cd3873e624c0d03aed52e249093882a3755407453e',
    'SR2-C1': '3f08bfbd8933212650543630ef200777a02b391710ea4d0af7802f2a29f87a0d',
    'SR2-C2': 'b1fffa4beb5ed8946cff71986ba6be102bae7ee65a84d566ad08e3303f8a78e4',
    'SR2-C3': 'aeb2923e9c014b32138f944e3dd269ff3705858c380f8d0e7d4d382b56ce6886',
    'SR2-C4': 'c4558478f46bf141d581976127f41419237dba590af3580962e513d1a010fbba',
    'SR2-D': 'f3b7015e4e2ee8b3787c8c4857f92409b55a63f1f8a73e4d916aa5e07289d945',
    'SRA1-C1': '1ebec4c77a5dd481611c41ac1b774df82e6df3db23ae098715b0e263b8e74a9c',
    'SRA1-C2': '5c8d97cd1e2a86ab5b41b8dea1ad4a6518062d41d53ae7114f7d1c4de0c63294',
    'SRA1-C3': '724d8ac9c09be45c5efdc6845c21f3079bfd1a19a1030ba9dd57cff1ac2942d7',
    'SRA1-C4': '5fb3025e76ad673f695ac68ad0f6074a54bf246e6bb6b087d3acab41cb22a87e',
    'SRA1-D': '9402a16fd11e4f3bafd4cbfb2a26221276606d20690de2991c5dcfa0aa4f1e03',
    'SRA2-C1': 'faaa89bf24b79250639eb24228b42444387890cf82f50d78a45c3f296b22e93e',
    'SRA2-C2': '6ac3c13f03b02a2167167edefc9b565e627c8773acf8d5030be5e98d95be74de',
    'SRA2-C3': 'fa20e12f83f4c7686ac106b2f2a1223adf94a0e3b3db4da8f0ccc263697f334f',
    'SRA2-C4': '76ba1aab9937d952ebcaffafd3db9e9d45ab2f8974d3f7e210589d439d1a0d54',
    'SRA2-D': '5885ed123c6ebbaf819634051a5b777a5fd3e0fc916cc12631c96b30aa5204b9',
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

        dataset_generator = dataset.generate(size=10, f=2000, features=['sourcemap', 'loc', 'f'], split='training')

        data_sample = next(dataset_generator)

    And finally, plot the results:

    .. code-block:: python

        import acoular as ac
        import matplotlib.pyplot as plt
        import numpy as np

        extent = dataset.config.grid.extent

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

    def __init__(
        self,
        srir_dir=None,
        scenario='A1',
        ref_mic_index=63,
        mode='welch',
        mic_sig_noise=True,
        random_signal_length=False,
        signal_length=5,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        remote_args=None,
        config=None,
    ):
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
        remote_args : dict, optional
            Dictionary of keyword arguments passed to the remote actors when using Ray for parallelization. Defaults to None.
        config : DatasetMIRACLEConfig, optional
            DatasetMIRACLEConfig object. Default is None, which creates a new DatasetMIRACLEConfig object.
        """
        if config is None:
            config = DatasetMIRACLEConfig(
                mode=mode,
                random_signal_length=random_signal_length,
                signal_length=signal_length,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                srir_dir=srir_dir,
                scenario=scenario,
                ref_mic_index=ref_mic_index,
                mic_sig_noise=mic_sig_noise,
            )
        super().__init__(tasks=tasks, remote_args=remote_args, config=config)


MIRACLE_SCENARIOS = ['A1', 'D1', 'A2', 'R2']
SRIRACHA_SCENARIOS = [
    'SR1',
    'SR1-C1',
    'SR1-C2',
    'SR1-C3',
    'SR1-C4',
    'SR1-D',
    'SR2',
    'SR2-C1',
    'SR2-C2',
    'SR2-C3',
    'SR2-C4',
    'SR2-D',
    'SRA1',
    'SRA1-C1',
    'SRA1-C2',
    'SRA1-C3',
    'SRA1-C4',
    'SRA1-D',
    'SRA2',
    'SRA2-C1',
    'SRA2-C2',
    'SRA2-C3',
    'SRA2-C4',
    'SRA2-D',
]


class DatasetMIRACLEConfig(DatasetSyntheticConfig):
    """Configuration class for the DatasetMIRACLE dataset."""

    srir_dir = Either(Instance(Path), Str, None)
    scenario = Either(MIRACLE_SCENARIOS, default='A1', desc='experimental configuration')
    filename = Property()
    _filename = Str
    ref_mic_index = Int(63, desc='reference microphone index (default: index of the centermost mic)')
    mic_pos_noise = Enum(False, desc='apply positional noise to microphone geometry')
    snap_to_grid = Enum(True, desc='snap source positions to measured grid')
    fs = Enum(32000, desc='sampling frequency')
    fft_params = Dict(
        {'block_size': 1024, 'overlap': '50%', 'window': 'Hanning', 'precision': 'complex64'},
        desc='FFT parameters',
    )

    def _get_filename(self):
        return self._filename

    def set_filename(self):
        """Set the filename of the SRIR file according to the scenario and srir_dir."""
        if link_address.get(self.scenario) is not None:
            self._filename = pooch.retrieve(
                url=link_address[self.scenario],
                fname=self.scenario + '.h5',
                path=self.srir_dir,
                known_hash=file_hash.get(self.scenario),
                progressbar=True,
            )
        else:
            # check if available locally
            if self.srir_dir is not None:
                # check if srir dir exists
                if not Path(self.srir_dir).is_dir():
                    msg = f'srir_dir {self.srir_dir} does not exist.'
                    raise ValueError(msg)

                local_path = Path(self.srir_dir) / (self.scenario + '.h5')
                if local_path.is_file():
                    self._filename = str(local_path)
                    return
                msg = f'File {local_path} does not exist.'
                raise ValueError(msg)
            msg = f'Invalid scenario {self.scenario}.'
            raise ValueError(msg)

    @observe(
        'mode, signal_length, max_nsources, mic_sig_noise, fft_params.items, scenario, ref_mic_index, filename',
        post_init=True,
    )
    def recreate_acoular_pipeline(self, event):
        self.create_acoular_pipeline()

    def create_acoular_pipeline(self):
        self.set_filename()
        self.env = self.create_env()
        self.mics = self.create_mics()
        self.noisy_mics = self.mics
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
        if self.mic_sig_noise:
            sampler[5] = self.mic_noise_sampler
        if self.random_signal_length:
            sampler[6] = self.signal_length_sampler
        return sampler

    def create_mics(self):
        with h5.File(self.filename, 'r') as file:
            pos_total = file['data/location/receiver'][()].T
        return ac.MicGeom(pos_total=pos_total)

    def create_env(self):
        with h5.File(self.filename, 'r') as file:
            c = np.mean(file['metadata/c0'][()])
        return ac.Environment(c=c)

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

    def create_steer(self):
        with h5.File(self.filename, 'r') as file:
            return ac.SteeringVector(
                steer_type='true level',
                mics=self.mics,
                grid=self.grid,
                env=self.env,
                ref=file['data/location/receiver'][self.ref_mic_index],
            )

    def create_grid(self):
        ap = self.mics.aperture
        with h5.File(self.filename, 'r') as file:
            z = file['data/location/source'][0, -1]
        return ac.RectGrid(y_min=-0.5 * ap, y_max=0.5 * ap, x_min=-0.5 * ap, x_max=0.5 * ap, z=z, increment=1 / 63 * ap)

    def create_source_grid(self):
        with h5.File(self.filename, 'r') as file:
            gpos_file = file['data/location/source'][()].T
        return ac.ImportGrid(pos=gpos_file)

    @staticmethod
    def _prepare_ir(sampler, mics, freq_data, filename, loc, ref_mic, domain='frequency'):
        fftfreq = freq_data.fftfreq()
        nfft = freq_data.fftfreq().shape[0]
        nsources = loc.shape[1]
        num_mics = mics.num_mics

        # we don't use a chunk cache here, since we access the data only once
        with h5.File(filename, 'r', rdcc_nbytes=0) as file:
            # finding the SRIR matching the location
            if domain == 'frequency':
                transfer = np.empty((nfft, num_mics, nsources), dtype=complex)
            loc_array = file['data/location/source'][()].T
            h_norm = np.zeros(nsources)
            irs = []
            for i in range(nsources):
                distances = np.linalg.norm(loc_array - loc[:, i][:, np.newaxis], axis=0)
                ir_idx = np.argmin(distances)
                assert distances[ir_idx] < 1e-6  # Ensure it's a close match
                ir = file['data/impulse_response'][ir_idx]
                irs.append(ir)
                h_norm[i] = np.sum(ir[ref_mic] ** 2)
                if domain == 'frequency':
                    transfer[:, :, i] = calc_transfer(ir, freq_data.sample_freq, freq_data.block_size, fftfreq)
        if domain == 'frequency':
            # calc transfer norm
            transfer /= np.sqrt(h_norm[np.newaxis, np.newaxis, :])
            return transfer
        # normalize irs
        irs = np.array(irs).transpose(1, 0, 2)  # mics x sources x time
        irs /= np.sqrt(h_norm[np.newaxis, :, np.newaxis])
        return irs

    @staticmethod
    def _prepare_ir_kernel(ir, sources, ref_sources, ref_mic):
        for i, src in enumerate(sources):
            src.kernel = ir[:, i, :].T
        for i, src in enumerate(ref_sources):
            src.kernel = ir[ref_mic, i, :].T[:, np.newaxis]

    @staticmethod
    def calc_analytic_prepare_func(sampler, mics, freq_data, filename, ref_mic):
        cf = DatasetSyntheticConfig
        cfm = DatasetMIRACLEConfig
        mics = cf._prepare_mics(sampler, mics)
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        H = cfm._prepare_ir(sampler, mics, freq_data, filename, loc, ref_mic)
        noise_prms_sq = cf._prepare_noise_params(sampler, prms_sq)
        cf._prepare_spectra_wishart(
            mics,
            freq_data,
            loc,
            prms_sq,
            source_seeds,
            noise_prms_sq,
            num_samples,
            custom_transfer=H,
        )
        return {
            'loc': loc,
            'prms_sq': prms_sq,
            'h_sq': np.real(H[:, ref_mic, :] * H[:, ref_mic, :].conj()),
        }

    @staticmethod
    def calc_welch_prepare_func(
        sampler, mics, beamformer, sources, fft_spectra, fft_obs_spectra, obs, filename, ref_mic
    ):
        cf = DatasetSyntheticConfig
        cfm = DatasetMIRACLEConfig
        freq_data = beamformer.freq_data
        mics = cf._prepare_mics(sampler, mics)
        loc, prms_sq, source_seeds, num_samples = cf._prepare_source_params(sampler, freq_data.sample_freq)
        ir = cfm._prepare_ir(sampler, mics, freq_data, filename, loc, ref_mic, 'time')
        subset_sources = cf._prepare_sources_welch(sources, loc, mics)
        signals = cf._prepare_signals_welch(prms_sq, subset_sources, num_samples, source_seeds)
        num_samples = signals[0].num_samples
        cf._prepare_spectra_welch(subset_sources, freq_data, fft_spectra, fft_obs_spectra, obs)
        cf._prepare_noise_welch(sampler, prms_sq, source_seeds[0] + 1000, freq_data, num_samples, mics)
        cfm._prepare_ir_kernel(ir, freq_data.source.sources, fft_obs_spectra.source.sources, ref_mic)
        # calc ref transfer for prms_sq_f
        H_ref = calc_transfer(ir[ref_mic, :, :], freq_data.sample_freq, freq_data.block_size, freq_data.fftfreq())
        return {
            'loc': loc,
            'prms_sq': prms_sq,
            'h_sq': np.real(H_ref * H_ref.conj()),
        }

    def get_prepare_func(self):
        if self.mode == 'welch':
            prepare_func = partial(
                self.calc_welch_prepare_func,
                beamformer=self.beamformer,
                mics=self.mics,
                sources=self.sources,
                fft_spectra=self.fft_spectra,
                fft_obs_spectra=self.fft_obs_spectra,
                obs=self.obs,
                filename=self.filename,
                ref_mic=self.ref_mic_index,
            )
        else:
            prepare_func = partial(
                self.calc_analytic_prepare_func,
                mics=self.mics,
                freq_data=self.freq_data,
                filename=self.filename,
                ref_mic=self.ref_mic_index,
            )
        return prepare_func


class DatasetSRIRACHA(DatasetMIRACLE):
    """A microphone array dataset generator using experimentally measured data from the SRIRACHA dataset."""

    def __init__(
        self,
        scenario,
        srir_dir=None,
        ref_mic_index=63,
        mode='welch',
        mic_sig_noise=True,
        random_signal_length=False,
        signal_length=5,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        config=None,
    ):
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
            config = DatasetSRIRACHAConfig(
                mode=mode,
                random_signal_length=random_signal_length,
                signal_length=signal_length,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                srir_dir=srir_dir,
                scenario=scenario,
                ref_mic_index=ref_mic_index,
                mic_sig_noise=mic_sig_noise,
            )
        super().__init__(tasks=tasks, config=config)


class DatasetSRIRACHAConfig(DatasetMIRACLEConfig):
    """Configuration class for the DatasetSRIRACHA dataset."""

    scenario = Either(*SRIRACHA_SCENARIOS, desc='experimental configuration')
