from copy import deepcopy
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

import acoular as ac
import h5py as h5
import numpy as np
from tqdm import tqdm
from traits.api import Dict, Either, Enum, Instance, Int, Property, Str, on_trait_change

from acoupipe.config import TF_FLAG
from acoupipe.datasets.base import DatasetBase
from acoupipe.datasets.features import BaseFeatureCollection
from acoupipe.datasets.synthetic import Dataset1Config, Dataset1FeatureCollectionBuilder
from acoupipe.datasets.utils import blockwise_transfer, get_uncorrelated_noise_source_recursively, tqdm_hook

link_address = {
"A1" : "https://depositonce.tu-berlin.de/bitstreams/76798c39-5ced-43f4-be09-7c0c30abdd81/download",
"A2" : "https://depositonce.tu-berlin.de/bitstreams/27b81b48-24a7-4db8-acc0-5dc085c94caf/download",
"D1" : "https://depositonce.tu-berlin.de/bitstreams/f812254f-185a-4e92-aff7-9daaebfe6976/download",
"R2" : "https://depositonce.tu-berlin.de/bitstreams/2b016da5-073d-4bb1-a05d-d1337aba00a1/download",
}

class DatasetMIRACLE(DatasetBase):

    def __init__(self, mode="analytic", mic_sig_noise=True, signal_length=5,min_nsources=1, max_nsources=10,
                srir_dir=None, exp_config="A1", ref_mic_index=63, tasks=1, config=None,
                ):
        if srir_dir is None:
            srir_dir = Path(ac.config.cache_dir).absolute()
        if config is None:
            config = DatasetMIRACLEConfig(
                mode=mode, signal_length=signal_length,min_nsources=min_nsources, max_nsources=max_nsources,
                srir_dir=srir_dir, exp_config=exp_config, ref_mic_index=ref_mic_index, mic_sig_noise=mic_sig_noise)
        self.config = config
        super().__init__(tasks=tasks, config=config)

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

        builder = MIRACLEFeatureCollectionBuilder(
            feature_collection = BaseFeatureCollection(),
            tdim = int(self.config.signal_length*self.config.fs),
            mdim = self.config.mics.num_mics,
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
                return builder.build()
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
                self.config.freq_data, f, num, ref_mic = self.config.ref_mic_index)
        if "source_strength_estimated" in features:
            if self.config.mode == "welch":
                builder.add_source_strength_estimated(
                self.config.fft_obs_spectra, f, num, ref_mic = self.config.ref_mic_index)
            else:
                builder.add_source_strength_estimated(
                    self.config.freq_data, f, num, ref_mic = self.config.ref_mic_index)
        if "noise_strength_analytic" in features:
            builder.add_noise_strength_analytic(self.config.freq_data, f, num)
        if "noise_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config.fft_spectra
            else:
                freq_data = self.config.freq_data
            builder.add_noise_strength_estimated(freq_data, f, num)
        if "f" in features:
            builder.add_f(self.config.freq_data.fftfreq(), f, num)
        if "num" in features:
            builder.add_num(num)
        return builder.build()


class MIRACLEFeatureCollectionBuilder(Dataset1FeatureCollectionBuilder):

    def add_source_strength_analytic(self, freq_data, f, num, ref_mic):
        from acoupipe.datasets.features import AnalyticSourceStrengthFeature
        calc_strength = AnalyticSourceStrengthFeature(
            freq_data=freq_data, f=f, num=num, ref_mic=ref_mic).get_feature_func()
        self.feature_collection.add_feature_func(calc_strength)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"source_strength_analytic" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"source_strength_analytic" : (self.fdim,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"source_strength_analytic" : "float32"})

    def add_source_strength_estimated(self, freq_data, f, num, ref_mic) :
        from acoupipe.datasets.features import EstimatedSourceStrengthFeature
        calc_strength = EstimatedSourceStrengthFeature(
            freq_data=freq_data, f=f, num=num, ref_mic=ref_mic).get_feature_func()
        self.feature_collection.add_feature_func(calc_strength)
        if TF_FLAG:
            from acoupipe.writer import float_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update(
                {"source_strength_estimated" : float_list_feature})
            self.feature_collection.feature_tf_shape_mapper.update(
                {"source_strength_estimated" : (self.fdim,None)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                {"source_strength_estimated" : "float32"})


class DatasetMIRACLEConfig(Dataset1Config):

    srir_dir = Instance(Path)
    exp_config = Either("A1","D1","A2","R2", default="A1", desc="experimental configuration")
    filename = Property()
    _filename = Str
    ref_mic_index = Int(63, desc="reference microphone index (default: index of the centermost mic)")
    mic_pos_noise = Enum(False, desc="apply positional noise to microphone geometry")
    snap_to_grid = Enum(True, desc="snap source positions to measured grid")
    fs = Enum(32000, desc="sampling frequency")
    fft_params = Dict({
                    "block_size" : 512,
                    "overlap" : "50%",
                    "window" : "Hanning",
                    "precision" : "complex64"},
                desc="FFT parameters")

    def _get_filename(self):
        return self._filename

    @on_trait_change("srir_dir, exp_config")
    def set_filename(self):
        if self.srir_dir is not None:
            file = Path(self.srir_dir / f"{self.exp_config}").with_suffix(".h5")
            if not file.exists():
                if not self.srir_dir.exists():
                    self.srir_dir.mkdir(parents=True)
                print(f"Downloading SRIR file {self.exp_config+'.h5'} to {self.srir_dir}...")
                with tqdm(unit = "B", unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
                    file, headers = urlretrieve(
                        link_address[self.exp_config], file, reporthook = tqdm_hook(t), data = None)
                for name, value in headers.items():
                    print(f"{name}: {value}")
            self._filename = str(file)

    @on_trait_change("mode, signal_length, max_nsources, mic_sig_noise, fft_params, exp_config, ref_mic_index, filename")
    def create_acoular_pipeline(self):
        if Path(self.filename).is_file():
            self.env = self._create_env()
            self.mics = self._create_mics()
            self.noisy_mics = self.mics
            self.grid = self._create_grid()
            self.source_grid = self._create_source_grid()
            self.steer = self._create_steer()
            self.obs = self._create_obs()
            self.signals = self._create_signals()
            self.sources = self._create_sources()
            self.source_steer = self._create_source_steer()
            self.mic_noise_signal = self._create_mic_noise_signal()
            self.mic_noise_source = self._create_mic_noise_source()
            self.freq_data = self._create_freq_data()
            self.fft_spectra = self._create_fft_spectra()
            self.fft_obs_spectra = self._create_fft_obs_spectra()
            self.beamformer = self._create_beamformer()

    def create_sampler(self):
        self.location_sampler = self._create_location_sampler()
        self.signal_seed_sampler = self._create_signal_seed_sampler()
        self.rms_sampler = self._create_rms_sampler()
        self.nsources_sampler = self._create_nsources_sampler()
        self.mic_noise_sampler = self._create_mic_noise_sampler()

    def get_sampler(self):
        self.create_sampler()
        sampler = {
            2 : self.signal_seed_sampler,
            3 : self.rms_sampler,
            4 : self.location_sampler,
            }

        if self.max_nsources != self.min_nsources:
            sampler[0] = self.nsources_sampler
        if self.mic_sig_noise:
            sampler[5] = self.mic_noise_sampler
        return sampler

    def _create_mics(self):
        with h5.File(self.filename, "r") as file:
            mpos_tot = file["data/location/receiver"][()].T
        return ac.MicGeom(mpos_tot = mpos_tot)

    def _create_env(self):
        with h5.File(self.filename, "r") as file:
            c = np.mean(file["metadata/c0"][()])
        return ac.Environment(c=c)

    def _create_sources(self):
        sources = []
        for signal in self.signals:
            sources.append(
                ac.PointSourceConvolve(
                    signal=signal,
                    mics=self.noisy_mics,
                    env=self.env,
                    )
            )
        return sources

    def _create_steer(self):
        with h5.File(self.filename, "r") as file:
            return ac.SteeringVector(
                steer_type="true level",
                mics=self.mics,
                grid=self.grid,
                env=self.env,
                ref=file["data/location/receiver"][self.ref_mic_index],
                    )

    def _create_grid(self):
        ap = self.mics.aperture
        with h5.File(self.filename, "r") as file:
            z = file["data/location/source"][0,-1]
        return ac.RectGrid(y_min=-0.5*ap, y_max=0.5*ap, x_min=-0.5*ap, x_max=0.5*ap,
                                    z=z, increment=1/63*ap)

    def _create_source_grid(self):
        with h5.File(self.filename, "r") as file:
            gpos_file = file["data/location/source"][()].T
        return ac.ImportGrid(gpos_file=gpos_file)

    @staticmethod
    def calc_analytic_prepare_func(sampler, beamformer, filename, ref_mic):
        with h5.File(filename, "r") as file:
            seed_sampler = sampler.get(2)
            rms_sampler = sampler.get(3)
            loc_sampler = sampler.get(4)
            noise_sampler = sampler.get(5)
            freq_data = beamformer.freq_data
            nfft = freq_data.fftfreq().shape[0]
            # sample parameters
            loc = loc_sampler.target
            nsources = loc.shape[1]
            # finding the SRIR matching the location
            transfer = np.empty((nfft, beamformer.steer.mics.num_mics,nsources), dtype=complex)
            for i in range(nsources):
                ir_idx = np.where( np.sum(loc_sampler.grid.gpos - loc[:,i][:,np.newaxis],axis=0) == 0)
                assert len(ir_idx) == 1
                tf = blockwise_transfer(
                    file["data/impulse_response"][ir_idx[0][0]], freq_data.block_size).T
                # normalize the transfer functions to match the prms at reference mic
                tf_pow = np.real(tf[:,ref_mic]*tf[:,ref_mic].conjugate()).sum(0) / nfft
                tf = tf / np.sqrt(tf_pow)
                transfer[:,:,i] = tf
            # adjust freq_data
            freq_data.custom_transfer = transfer
            freq_data.steer.grid = ac.ImportGrid(gpos_file=loc) # set source locations
            freq_data.seed=seed_sampler.target
            # change source strength
            prms_sq = rms_sampler.target[:nsources] # squared sound pressure RMS at reference position
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

    @staticmethod
    def calc_welch_prepare_func(sampler, beamformer, sources, fft_spectra, fft_obs_spectra, obs, filename, ref_mic):
        with h5.File(filename, "r") as file:
            # restore sampler and acoular objects
            seed_sampler = sampler.get(2)
            rms_sampler = sampler.get(3)
            loc_sampler = sampler.get(4)
            noise_sampler = sampler.get(5)
            freq_data = beamformer.freq_data
            # sample parameters
            loc = loc_sampler.target
            nsources = loc.shape[1]
            prms_sq = rms_sampler.target[:nsources] # squared sound pressure RMS at reference position
            # apply parameters
            mic_noise = get_uncorrelated_noise_source_recursively(freq_data.source)
            if mic_noise:
                mic_noise_signal = mic_noise[0].signal
                if noise_sampler is not None:
                    noise_signal_ratio = noise_sampler.target # normalized noise variance
                    noise_prms_sq = prms_sq.sum()*noise_signal_ratio
                    mic_noise_signal.rms = np.sqrt(noise_prms_sq)
                    mic_noise_signal.seed = seed_sampler.target+1000
            subset_sources = sources[:nsources]
            for i,src in enumerate(subset_sources):
                ir_idx = np.where( np.sum(loc_sampler.grid.gpos - loc[:,i][:,np.newaxis],axis=0) == 0)
                assert len(ir_idx) == 1
                kernel = file["data/impulse_response"][ir_idx[0][0]].T
                # normalize energy
                kernel /=  np.sqrt(np.sum(kernel[:,ref_mic]**2))
                src.kernel = kernel
                src.signal.seed = seed_sampler.target+i
                src.signal.rms = np.sqrt(prms_sq[i])
            freq_data.source.sources = subset_sources # apply subset of sources
            fft_spectra.source = freq_data.source # only for spectrogram feature
            # update observation point
            obs_sources = deepcopy(subset_sources)
            for src in obs_sources:
                src.mics = obs
                src.kernel = src.kernel[:,ref_mic][:,np.newaxis]
            fft_obs_spectra.source = ac.SourceMixer(sources=obs_sources)
        return {}

    def get_prepare_func(self):
        if self.mode == "welch":
            prepare_func = partial(
            self.calc_welch_prepare_func,
            beamformer=self.beamformer,
            sources=self.sources,
            fft_spectra=self.fft_spectra,
            fft_obs_spectra=self.fft_obs_spectra,
            obs = self.obs,
            filename=self.filename,
            ref_mic=self.ref_mic_index)
        else:
            prepare_func = partial(
            self.calc_analytic_prepare_func,
            beamformer=self.beamformer,
            filename=self.filename,
            ref_mic=self.ref_mic_index)
        return prepare_func





