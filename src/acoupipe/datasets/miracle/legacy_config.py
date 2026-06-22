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
from acoupipe.datasets._shared.utils import (
    calc_transfer,
)
from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig

import h5py as h5
import numpy as np
from irdl import MiracleDataset
from traits.api import Dict, Either, Enum, Instance, Int, Property, Str, observe

_MIRACLE_SCENARIOS = ['A1', 'D1', 'A2', 'R2']


class DatasetMIRACLEConfig(DatasetSyntheticConfig):
    """Configuration class for the DatasetMIRACLE dataset."""

    srir_dir = Either(Instance(Path), Str, None)
    scenario = Either(_MIRACLE_SCENARIOS, default='A1', desc='experimental configuration')
    dataset_split = Either(None, 'C1', 'C2', 'C3', 'C4', default=None, desc='artificial dataset split')
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
        """Resolve the SRIR file path, downloading via :mod:`irdl` if necessary."""
        output_format = 'hdf5' if self.dataset_split else 'raw'

        self._filename = str(
            MiracleDataset.get(
                scenario=self.scenario,
                dataset_split=self.dataset_split,
                cache_dir=self.srir_dir,
                output_format=output_format,
            )
        )

    @observe(
        'mode, signal_length, max_nsources, mic_sig_noise, fft_params.items, scenario, dataset_split, ref_mic_index, filename',
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
            pos_total = self._read_receiver_positions(file)
        return ac.MicGeom(pos_total=pos_total)

    def create_env(self):
        with h5.File(self.filename, 'r') as file:
            c = np.mean(self._read_speed_of_sound(file))
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
                ref=self._read_receiver_positions(file)[:, self.ref_mic_index],
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
    def _read_receiver_positions(file):
        receiver = np.asarray(file['data/location/receiver'][()])
        if receiver.ndim == 3 and receiver.shape[-1] == 1:
            receiver = receiver[..., 0]
        return receiver.T

    @staticmethod
    def _read_speed_of_sound(file):
        if 'metadata/c0' in file:
            return file['metadata/c0'][()]
        msg = "Could not find speed of sound in 'metadata/c0'."
        raise KeyError(msg)

    @staticmethod
    def _read_impulse_response(file, index):
        ir = np.asarray(file['data/impulse_response'][index])
        if ir.ndim == 3 and ir.shape[-1] == 1:
            ir = ir[..., 0]
        return ir

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
                ir = DatasetMIRACLEConfig._read_impulse_response(file, ir_idx)
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


__all__ = ['DatasetMIRACLEConfig']
