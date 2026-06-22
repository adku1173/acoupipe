"""Legacy Synthetic ISM dataset configuration."""

from functools import partial

import acoular as ac
from acoupipe.datasets._shared.utils import calc_transfer
from acoupipe.datasets.ism.ir import get_ir
from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig

import numpy as np
from traits.api import Float, List


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


__all__ = ['DatasetSyntheticISMConfig']
