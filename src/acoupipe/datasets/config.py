from functools import partial

import acoular as ac
import numpy as np
from traits.api import Enum, Instance

from acoupipe.datasets.base import ConfigBase
from acoupipe.datasets.collection import SyntheticCollectionBuilderAnalytic, SyntheticCollectionBuilderWelch
from acoupipe.datasets.setup import SyntheticSamplerSetup, SyntheticSetupAnalytic, SyntheticSetupWelch


class DatasetSyntheticConfigAnalytic(ConfigBase):
    """
    Default Configuration class for synthetic data simulation.

    Attributes
    ----------
    msm_setup : SyntheticSetupAnalytic
        SyntheticSetupAnalytic object.
    sampler_setup : SyntheticSamplerSetup
        SyntheticSamplerSetup object.
    mode : Enum
        type of PowerSpectra calculation method.
    """

    msm_setup = Instance(SyntheticSetupAnalytic, desc="SyntheticSetupAnalytic object.")
    sampler_setup = Instance(SyntheticSamplerSetup, desc="SyntheticSamplerSetup object.")
    mode = Enum(("analytic", "wishart"), default="wishart",
                            desc="type of PowerSpectra calculation method.")

    def __init__(self, msm_setup=None, sampler_setup=None, **kwargs):
        if msm_setup is None:
            setup_class_attr = SyntheticSetupAnalytic.class_traits().keys()
            setup_kwargs = {key: kwargs[key] for key in setup_class_attr if key in kwargs}
            msm_setup = SyntheticSetupAnalytic(**setup_kwargs)

        if sampler_setup is None:
            sampler_setup_class_attr = SyntheticSamplerSetup.class_traits().keys()
            setup_kwargs = {key: kwargs[key] for key in sampler_setup_class_attr if key in kwargs}
            sampler_setup = SyntheticSamplerSetup(**setup_kwargs)

        sampler_setup.msm_setup = msm_setup
        kwargs.setdefault("mode", "wishart")

        # handle unrecognized kwargs
        all_keys = set(SyntheticSetupAnalytic.class_traits().keys()) | set(SyntheticSamplerSetup.class_traits().keys())
        unrecognized_keys = set(kwargs.keys()) - all_keys
        if unrecognized_keys:
            raise ValueError(f"Unrecognized keys in kwargs: {unrecognized_keys}")

        super().__init__(msm_setup=msm_setup, sampler_setup=sampler_setup)

    def _connect_objects(self, sampler, msm_setup):
        msm_setup.freq_data.steer = msm_setup.steer

    def _apply_new_mic_pos(self,sampler, msm_setup):
        msm_setup.freq_data.steer.mics = sampler[1].target

    def _apply_new_seeds(self,sampler, msm_setup):
        seed_sampler = sampler.get(2)
        if seed_sampler is not None:
            msm_setup.freq_data.seed = seed_sampler.target

    def _apply_new_loc(self,sampler, msm_setup):
        loc = sampler[4].target
        msm_setup.freq_data.steer.grid = ac.ImportGrid(gpos_file=loc)

    def _apply_new_signal_length(self,sampler, msm_setup):
        signal_length_sampler = sampler.get(6)
        if signal_length_sampler is not None:
            msm_setup.freq_data.numsamples = signal_length_sampler.target*msm_setup.freq_data.sample_freq

    def _apply_new_source_strength(self,sampler, msm_setup):
        nfft = msm_setup.freq_data.fftfreq().shape[0]
        nsources = sampler[4].nsources
        prms_sq = sampler[3].target[:nsources]**2
        prms_sq_per_freq = prms_sq / nfft #prms_sq_per_freq
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
        numsampler = self.sampler_setup.numsampler
        builder = SyntheticCollectionBuilderAnalytic(msm_setup=self.msm_setup, f=f, num=num)

        # add prepare function
        builder.add_custom(self.get_prepare_func())
        builder.add_seeds(numsampler)
        builder.add_idx()

        # add feature functions
        if "time_data" in features:
            raise ValueError("time_data feature is not possible with modes ['analytic', 'wishart'].")
        if "spectrogram" in features:
            raise ValueError("spectrogram feature is not possible with modes ['analytic', 'wishart'].")
        if "csm" in features:
            builder.add_csm()
        if "csmtriu" in features:
            builder.add_csmtriu()
        if "eigmode" in features:
            builder.add_eigmode()
        if "sourcemap" in features:
            builder.add_sourcemap()
        if "loc" in features:
            builder.add_loc()
        if "source_strength_analytic" in features:
            builder.add_source_strength_analytic( # TODO: handle in prepare function! -> particularly unsuited for welch
                self.config._freq_data, f, num, steer=self.config._source_steer)
        if "source_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_ref_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_source_strength_estimated(freq_data, f, num)
        if "noise_strength_analytic" in features:
            builder.add_noise_strength_analytic(self.config._freq_data, f, num)
        if "noise_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_noise_strength_estimated(freq_data, f, num)
        if "targetmap_analytic" in features:
            builder.add_targetmap(self.config._freq_data, f, num, self.config._source_steer,
                ref_mic=None, strength_type="analytic", grid=self.config.grid)
        if "targetmap_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_ref_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_targetmap(freq_data, f, num, self.config._source_steer,
                ref_mic=None, strength_type="estimated", grid=self.config.grid)
        if "f" in features:
            builder.add_f()
        if "num" in features:
            builder.add_num()
        #TODO: simply add a function that kicks off all features not part of the
        # feature list (this way one can add already during the prepare function step)
        # add function must become a flag that excludes feature func when adding
        return builder.feature_collection


class DatasetSyntheticConfigWelch(ConfigBase):
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
    sampler_setup = Instance(SyntheticSamplerSetup, desc="SyntheticSamplerSetup object.")

    def __init__(self, msm_setup=None, sampler_setup=None, mode="welch", **kwargs):
        if msm_setup is None:
            setup_class_attr = SyntheticSetupWelch.class_traits().keys()
            setup_kwargs = {key: kwargs[key] for key in setup_class_attr if key in kwargs}
            msm_setup = SyntheticSetupWelch(**setup_kwargs)

        if sampler_setup is None:
            sampler_setup_class_attr = SyntheticSamplerSetup.class_traits().keys()
            setup_kwargs = {key: kwargs[key] for key in sampler_setup_class_attr if key in kwargs}
            sampler_setup = SyntheticSamplerSetup(**setup_kwargs)

        sampler_setup.msm_setup = msm_setup

        # handle unrecognized kwargs
        all_keys = set(SyntheticSetupWelch.class_traits().keys()) | set(SyntheticSamplerSetup.class_traits().keys())
        unrecognized_keys = set(kwargs.keys()) - all_keys
        if unrecognized_keys:
            raise ValueError(f"Unrecognized keys in kwargs: {unrecognized_keys}")

        super().__init__(msm_setup=msm_setup, sampler_setup=sampler_setup)

    def _connect_objects(self, sampler, msm_setup):
        nsources = sampler[4].nsources
        if self.sampler_setup.mic_sig_noise:
            mix = ac.SourceMixer(sources=msm_setup.sources[:nsources] + [msm_setup.sources[-1]])
        else:
            mix = ac.SourceMixer(sources=msm_setup.sources[:nsources])
        self.msm_setup.freq_data.source = mix
        self.msm_setup.fft_spectra.source = mix

    def _apply_new_mic_pos(self, sampler, msm_setup):
        for src in msm_setup.sources:
            src.mics = sampler[1].target

    def _apply_new_loc(self, sampler, msm_setup):
        locations = sampler[4].target
        for i, loc in enumerate(locations.T):
            msm_setup.sources[i].loc = (loc[0], loc[1], loc[2])

    def _apply_new_seeds(self, sampler, msm_setup):
        seed = sampler.get(2).target
        for i in range(sampler[4].nsources):
            msm_setup.sources[i].signal.seed = seed + i
        if self.sampler_setup.mic_sig_noise:
            msm_setup.sources[-1].signal.seed = seed + 1000

    def _apply_new_signal_length(self, sampler, msm_setup):
        signal_length_sampler = sampler.get(6)
        t = signal_length_sampler.target
        fs = msm_setup.sources[0].signal.sample_freq
        for src in msm_setup.sources:
            src.signal.numsamples = int(t*fs)

    def _apply_new_source_strength(self, sampler, msm_setup):
        nsources = sampler[4].nsources # loc sampler
        prms_sq = sampler[3].target[:nsources]**2
        for i in range(nsources):
            r0 = np.linalg.norm(msm_setup.sources[i].loc-msm_setup.steer.ref)
            msm_setup.sources[i].signal.rms = np.sqrt(prms_sq[i])*r0
            #msm_setup.sources[i].signal.rms = np.sqrt(prms_sq[i])*source_steer.r0[i] # TODO: use r0 function for light weight source

    def _apply_new_mic_sig_noise(self, sampler, msm_setup):
        noise_sampler = sampler.get(5)
        if noise_sampler is not None:
            nsources = sampler[4].nsources # loc sampler
            prms_sq = sampler[3].target[:nsources]**2
            noise_signal_ratio = noise_sampler.target # normalized noise variance
            noise_prms_sq = prms_sq.sum()*noise_signal_ratio
            msm_setup.sources[-1].signal.rms = np.sqrt(noise_prms_sq)

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
        numsampler = self.sampler_setup.numsampler
        builder = SyntheticCollectionBuilderWelch(msm_setup=self.msm_setup, f=f, num=num)

        # add prepare function
        builder.add_custom(self.get_prepare_func())
        builder.add_seeds(numsampler)
        builder.add_idx()

        # add feature functions
        if "time_data" in features:
            raise ValueError("time_data feature is not possible with modes ['analytic', 'wishart'].")
        if "spectrogram" in features:
            raise ValueError("spectrogram feature is not possible with modes ['analytic', 'wishart'].")
        if "csm" in features:
            builder.add_csm()
        if "csmtriu" in features:
            builder.add_csmtriu()
        if "eigmode" in features:
            builder.add_eigmode()
        if "sourcemap" in features:
            builder.add_sourcemap()
        if "loc" in features:
            builder.add_loc() # TODO: handle in prepare function ! -> particularly unsuited for welch
        if "source_strength_analytic" in features:
            builder.add_source_strength_analytic( # TODO: handle in prepare function! -> particularly unsuited for welch
                self.config._freq_data, f, num, steer=self.config._source_steer)
        if "source_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_ref_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_source_strength_estimated(freq_data, f, num)
        if "noise_strength_analytic" in features:
            builder.add_noise_strength_analytic(self.config._freq_data, f, num)
        if "noise_strength_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_noise_strength_estimated(freq_data, f, num)
        if "targetmap_analytic" in features:
            builder.add_targetmap(self.config._freq_data, f, num, self.config._source_steer,
                ref_mic=None, strength_type="analytic", grid=self.config.grid)
        if "targetmap_estimated" in features:
            if self.config.mode == "welch":
                freq_data = self.config._fft_ref_spectra
            else:
                freq_data = self.config._freq_data
            builder.add_targetmap(freq_data, f, num, self.config._source_steer,
                ref_mic=None, strength_type="estimated", grid=self.config.grid)
        if "f" in features:
            builder.add_f()
        if "num" in features:
            builder.add_num()
        #TODO: simply add a function that kicks off all features not part of the
        # feature list (this way one can add already during the prepare function step)
        # add function must become a flag that excludes feature func when adding
        return builder.feature_collection


