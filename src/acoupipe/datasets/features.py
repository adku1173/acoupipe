from functools import partial

import acoular as ac
import numpy as np
from numpy import array, imag, newaxis, real, triu_indices
from numpy.linalg import eigh
from traits.api import Callable, Dict, Either, Enum, Float, HasPrivateTraits, Instance, Int, List, Property, Str

from acoupipe.config import TF_FLAG
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.datasets.utils import (
    blockwise_transfer,
    get_frequency_index_range,
    get_point_sources_recursively,
    get_uncorrelated_noise_source_recursively,
)


class BaseFeatureCatalog(HasPrivateTraits):
    """BaseFeatureCatalog base class for handling feature funcs.

    Attributes
    ----------
    name : str
        Name of the feature.

    """

    name = Str
    dtype = Callable

    def get_feature_func(self):
        """Will return a method depending on the class parameters."""
        return


class TimeDataFeature(BaseFeatureCatalog):
    """TimeDataFeature class for handling time data.

    Attributes
    ----------
    name : str
        Name of the feature (default='time_data').
    time_data : instance of class acoular.SamplesGenerator
        The source delivering the time data.
    """

    name = Str("time_data")
    time_data = Instance(ac.SamplesGenerator, desc="time data")

    def get_feature_func(self):
        """Return the callable for calculating the time data."""
        def _calc_time_data(sampler, time_data, name, dtype):
            return {
                name: dtype(ac.tools.return_result(time_data))}
        return partial(_calc_time_data,time_data=self.time_data, name=self.name, dtype=self.dtype)


class TargetmapFeature(BaseFeatureCatalog):

    name = Str("targetmap")
    freq_data = Instance(ac.BaseSpectra, desc="cross spectral matrix calculation class")
    f = Either(None, Float, List(Float), desc="frequency")
    num = Int
    steer = Instance(ac.SteeringVector, desc="steering vector object defining the path between the sources and the microphones")
    ref_mic = Either(Int, None, default=None, desc="reference microphone index")
    strength_type = Either("analytic", "estimated", desc="source strength type")
    grid = Instance(ac.Grid, desc="grid")

    @staticmethod
    def get_targetmap(sampler, grid, loc_callable, strength_callable, name):
        loc = list(loc_callable(sampler=None).values())[0]
        strength = list(strength_callable(sampler=None).values())[0]
        # create target map
        if type(grid) is ac.RectGrid:
            loc = loc[:2]
        elif type(grid) is ac.RectGrid3D:
            loc = loc[:3]
        else:
            raise NotImplementedError(f"Unknown grid type {type(grid)}.")
            # other grid types in Acoular do not provide an index method!
        target_map = np.zeros((strength.shape[0],)+grid.shape)
        for j in range(loc.shape[1]):
            index = grid.index(*loc[:, j])
            target_map[(slice(None),) + index] += strength[:,j]
        return {name: target_map}

    def get_feature_func(self):
        loc_callable = LocFeature(freq_data=self.freq_data).get_feature_func()
        if self.strength_type == "analytic":
            strength_callable = AnalyticSourceStrengthFeature(
                freq_data=self.freq_data, f=self.f, num=self.num, steer=self.steer, ref_mic=self.ref_mic).get_feature_func()
        else:
            strength_callable = EstimatedSourceStrengthFeature(
                freq_data=self.freq_data, f=self.f, num=self.num, ref_mic=self.ref_mic).get_feature_func()
        return partial(self.get_targetmap, grid=self.grid, loc_callable=loc_callable, strength_callable=strength_callable, name=self.name)


class SourcemapFeature(BaseFeatureCatalog):
    """SourcemapFeature class for handling the generation of sourcemaps obtained with microphone array methods.

    Attributes
    ----------
    name : str
        Name of the feature (default='sourcemap').
    beamformer : instance of class acoular.BeamformerBase
        The beamformer to calculate the sourcemap.
    f : float
        The center frequency or list of frequencies of the dataset. If None, all frequencies are included.
    num : integer
        Controls the width of the frequency bands considered; defaults to
        0 (single frequency line).

        ===  =====================
        num  frequency band width
        ===  =====================
        0    single frequency line
        1    octave band
        3    third-octave band
        n    1/n-octave band
        ===  =====================
    fidx : list of tuples
        List of tuples containing the start and end indices of the frequency bands to be considered. Is determined
        automatically from attr:`f` and attr:`num`.
    """

    name = Str("sourcemap")
    beamformer = Instance(ac.BeamformerBase, desc="beamformer")
    f = Either(None, Float, List(Float), desc="frequency")
    num = Int
    fidx = Property(desc="frequency indices")

    def _get_fidx(self):
        """Return a list of tuples containing the frequency indices."""
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                f = [self.f]
            else:
                f = self.f
            fidx = [get_frequency_index_range(
                self.beamformer.freq_data.fftfreq(), f_, self.num) for f_ in f]
        else:
            if self.num != 0:
                raise ValueError(
            'Frequencies "f" must be given if a fractional octave band is specified.')
            fidx = None
        return fidx

    def set_freq_limits(self):
        """Set the frequency limits of the beamformer so that the result is only calculated for necessary frequencies."""
        if self.beamformer.freq_data is not None:
            if self.fidx is not None:
                self.beamformer.freq_data.ind_low = min([f[0] for f in self.fidx])
                self.beamformer.freq_data.ind_high = max([f[1] for f in self.fidx])
            else:
                self.beamformer.freq_data.ind_low = 0
                self.beamformer.freq_data.ind_high = None

    @staticmethod
    def calc_beamformer1(sampler, beamformer, f, num, name):
        sm = array([beamformer.synthetic(freq,num=num) for freq in f])
        return {name: sm}

    @staticmethod
    def calc_beamformer2(sampler, beamformer, name):
        f = beamformer.freq_data.fftfreq()
        sm = array([beamformer.synthetic(freq,num=0) for freq in f])
        return {name: sm}

    def get_feature_func(self):
        """Return the callable for calculating the sourcemap."""
        self.set_freq_limits()
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                f = [self.f]
            else:
                f = self.f
            return partial(self.calc_beamformer1, beamformer=self.beamformer, f=f, num=self.num, name=self.name)
        else:
            return partial(self.calc_beamformer2, beamformer=self.beamformer, name=self.name)

class SpectraFeature(BaseFeatureCatalog):
    """Handles the calculation of features in the frequency domain.

    Attributes
    ----------
    name : str
        Name of the feature.
    freq_data : instance of class acoular.BaseSpectra
        The frequency data to calculate the feature for.
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)
    fidx : list of tuples
        List of tuples containing the start and end indices of the frequency bands to be considered.
    """

    freq_data = Instance(ac.BaseSpectra, desc="spectrogram")
    f = Either(None, Float, List(Float), desc="frequency")
    num = Int
    fidx = Property(desc="frequency indices")

    def _get_fidx(self):
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                f = [self.f]
            else:
                f = self.f
            fidx = [get_frequency_index_range(
                self.freq_data.fftfreq(), f_, self.num) for f_ in f]
        else:
            if self.num != 0:
                raise ValueError(
            'Frequencies "f" must be given if a fractional octave band is specified.')
            fidx = None
        return fidx

    def set_freq_limits(self):
        """Set the frequency limits of the spectra object so that the result is only calculated for necessary frequencies."""
        if self.freq_data is not None:
            if self.fidx is not None:
                self.freq_data.ind_low = min([f[0] for f in self.fidx])
                self.freq_data.ind_high = max([f[1] for f in self.fidx])
            else:
                self.freq_data.ind_low = 0
                self.freq_data.ind_high = None


class SpectrogramFeature(SpectraFeature):
    """SpectrogramFeature class for handling spectrogram features.

    Attributes
    ----------
    name : str
        Name of the feature (default='spectrogram').
    freq_data : instance of class acoular.RFFT
        The object which calculates the spectrogram data.
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)
    fidx : list of tuples
        List of tuples containing the start and end indices of the frequency bands to be considered.
    """

    name = Str("spectrogram")
    freq_data = Instance(ac.RFFT, desc="spectrogram")

    @staticmethod
    def calc_spectrogram1(sampler, freq_data, name):
        spectrogram = np.stack(
            list((_.copy() for _ in freq_data.result())),axis=0)
        return {name:spectrogram}

    @staticmethod
    def calc_spectrogram2(sampler, freq_data, fidx, name):
        spectrogram = np.stack(
            list((_.copy() for _ in freq_data.result())),axis=0)
        spectrogram = np.array(
                [spectrogram[:,indices[0]:indices[1]].sum(1) for indices in fidx],
                    dtype=complex).swapaxes(0,1)
        return {name:spectrogram}

    def get_feature_func(self):
        if self.fidx is None:
            return partial(self.calc_spectrogram1, freq_data = self.freq_data, name=self.name)
        else:
            return partial(self.calc_spectrogram2, freq_data = self.freq_data,fidx = self.fidx, name=self.name)


class CSMFeature(SpectraFeature):
    """CSMFeature class for handling cross-spectral matrix calculation.

    Attributes
    ----------
    name : str
        Name of the feature (default='csm').
    freq_data : instance of class acoular.PowerSpectra
        The object which calculates the cross-spectral matrix.
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)
    fidx : list of tuples
        List of tuples containing the start and end indices of the frequency bands to be considered.
    """

    name = Str("csm")
    freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix")

    @staticmethod
    def calc_csm1(sampler, freq_data, name):
        """Calculate the cross-spectral matrix (CSM) from time data.

        Parameters
        ----------
        freq_data : instance of class acoular.PowerSpectra
            power spectra to calculate the csm feature

        Returns
        -------
        numpy.array
            The complex-valued cross-spectral matrix with shape (numfreq, num_mics, num_mics).
        """
        return {name: freq_data.csm[:]}

    @staticmethod
    def calc_csm2(sampler, freq_data, fidx, name):
        """Calculate the cross-spectral matrix (CSM) from time data.

        Parameters
        ----------
        freq_data : instance of class acoular.PowerSpectra
            power spectra to calculate the csm feature
        fidx : list of tuples, optional
            list of tuples containing the start and end indices of the frequency bands to be considered,
            by default None


        Returns
        -------
        numpy.array
            The complex-valued cross-spectral matrix with shape (numfreq, num_mics, num_mics) with numfreq
            depending on the number of frequencies in fidx.
        """
        csm = freq_data.csm[:]
        csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)
        return {name :csm}

    def get_feature_func(self):
        """Return the callable for calculating the cross-spectral matrix."""
        self.set_freq_limits()
        if self.fidx is None:
            return partial(self.calc_csm1, freq_data = self.freq_data, name=self.name)
        else:
            return partial(self.calc_csm2, freq_data = self.freq_data,fidx = self.fidx, name=self.name)


class CSMtriuFeature(SpectraFeature):

    name = Str("csmtriu")
    freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix calculation class")

    @staticmethod
    def transform(csm):
        csmtriu_real = np.zeros(csm.shape)
        csm_triu_imag = np.zeros(csm.shape)
        num_mics = csm.shape[1]
        for i in range(csm.shape[0]):
            csmtriu_real[i][triu_indices(num_mics)] = real(csm[i])[triu_indices(num_mics)] # add real part at upper triangular matrix
            csm_triu_imag[i][triu_indices(num_mics)] = imag(csm[i])[triu_indices(num_mics)]
        return csmtriu_real + csm_triu_imag.transpose(0,2,1)

    @staticmethod
    def calc_csmtriu1(sampler, freq_data, name):
        """Calculate the cross-spectral matrix (CSM) from time data.

        Parameters
        ----------
        freq_data : instance of class acoular.PowerSpectra
            power spectra to calculate the csm feature

        Returns
        -------
        numpy.array
            The real-valued cross-spectral matrix with shape (numfreq, num_mics, num_mics).
        """
        return {name: CSMtriuFeature.transform(freq_data.csm[:])}

    @staticmethod
    def calc_csmtriu2(sampler, freq_data, fidx, name):
        """Calculate the cross-spectral matrix (CSM) from time data.

        Parameters
        ----------
        freq_data : instance of class acoular.PowerSpectra
            power spectra to calculate the csm feature
        fidx : list of tuples, optional
            list of tuples containing the start and end indices of the frequency bands to be considered,
            by default None


        Returns
        -------
        numpy.array
            The real-valued cross-spectral matrix with shape (numfreq, num_mics, num_mics) with numfreq
            depending on the number of frequencies in fidx.
        """
        csm = freq_data.csm[:]
        csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)
        return {name: CSMtriuFeature.transform(csm)}

    def get_feature_func(self):
        self.set_freq_limits()
        if self.fidx is None:
            return partial(self.calc_csmtriu1, freq_data = self.freq_data, name=self.name)
        else:
            return partial(self.calc_csmtriu2, freq_data = self.freq_data,fidx = self.fidx, name=self.name)


class EigmodeFeature(SpectraFeature):

        name = Str("eigmode")
        freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix calculation class")

        @staticmethod
        def transform(csm):
            eva, eve = eigh(csm)
            return eva[:,newaxis,:]*eve[:]

        @staticmethod
        def calc_eigmode1(sampler, freq_data, name):
            """Calculate the eigenvalue-scaled eigenvectors of the cross-spectral matrix (CSM) from time data.

            Parameters
            ----------
            freq_data : instance of class acoular.PowerSpectra
                power spectra to calculate the csm feature

            Returns
            -------
            numpy.array
                The eigenvalue scaled eigenvectors with shape (numfreq, num_mics, num_mics).
            """
            return {name: EigmodeFeature.transform(freq_data.csm[:])}

        @staticmethod
        def calc_eigmode2(sampler, freq_data, fidx, name):
            """Calculate the eigenvalue-scaled eigenvectors of the cross-spectral matrix (CSM) from time data.

            Parameters
            ----------
            freq_data : instance of class acoular.PowerSpectra
                power spectra to calculate the csm feature
            fidx : list of tuples, optional
                list of tuples containing the start and end indices of the frequency bands to be considered,
                by default None


            Returns
            -------
            numpy.array
                The eigenvalue scaled eigenvectors with shape (numfreq, num_mics, num_mics) with numfreq
                depending on the number of frequencies in fidx.
            """
            csm = freq_data.csm[:]
            csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)
            return {name: EigmodeFeature.transform(csm)}


        def get_feature_func(self):
            self.set_freq_limits()
            if self.fidx is None:
                return partial(self.calc_eigmode1, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_eigmode2, freq_data = self.freq_data,fidx = self.fidx, name=self.name)


class LocFeature(BaseFeatureCatalog):

    name = Str("loc")
    freq_data = Instance(ac.BaseSpectra, desc="cross spectral matrix calculation class")

    @staticmethod
    def calc_loc1(sampler, freq_data, name):
        sources = get_point_sources_recursively(freq_data.source)
        locs = np.array([src.loc for src in sources]).T
        return {name: locs}

    @staticmethod
    def calc_loc2(sampler, freq_data, name):
        return {name: freq_data.steer.grid.gpos_file}

    def get_feature_func(self):
        if isinstance(self.freq_data, PowerSpectraAnalytic):
            return partial(self.calc_loc2, freq_data = self.freq_data, name=self.name)
        elif isinstance(self.freq_data, ac.BaseSpectra):
            return partial(self.calc_loc1, freq_data = self.freq_data, name=self.name)
        else:
            raise NotImplementedError(f"Unknown freq_data type {self.freq_data.__class__.__name__}.")


class AnalyticSourceStrengthFeature(SpectraFeature):

    name = Str("source_strength_analytic")
    cross_strength = Enum(False) # can later be extended for sources' cross-power values
    freq_data = Instance(ac.BaseSpectra, desc="cross spectral matrix calculation class")
    steer = Instance(ac.SteeringVector, desc="steering vector object defining the path between the sources and the microphones")
    ref_mic = Either(Int, None, default=None, desc="reference microphone index")


    @staticmethod
    def calc_source_strength_analytic1_fullfreq(sampler, freq_data, steer, ref_mic, name):
        sources = get_point_sources_recursively(freq_data.source)
        nfft = freq_data.fftfreq().shape[0]
        strength = np.zeros((nfft, len(sources)))
        for j, source in enumerate(sources):
            if isinstance(source, ac.PointSourceConvolve):
                ir = source.kernel[:,ref_mic].copy()
                tf = blockwise_transfer(ir[np.newaxis], blocksize=freq_data.block_size).squeeze()
                strength[:,j] = np.real(tf*tf.conjugate())*source.signal.rms**2/nfft
            elif isinstance(source, ac.PointSource):
                if isinstance(source.signal, ac.WNoiseGenerator):
                    strength[:,j] = np.ones(nfft)*(source.signal.rms/steer.r0[j])**2/nfft
                else:
                    raise NotImplementedError(
                        f"Cannot handle source signal type {source.signal.__class__.__name__}.")
            else:
                raise NotImplementedError(
                    f"Cannot handle source type {source.__class__.__name__}.")
        return {name: strength}

    @staticmethod
    def calc_source_strength_analytic1_partfreq(sampler, freq_data, fidx, name, steer, ref_mic):
        strength = AnalyticSourceStrengthFeature.calc_source_strength_analytic1_fullfreq(sampler, freq_data, steer, ref_mic, name)[name]
        return {name: np.array([strength[indices[0]:indices[1]].sum(0) for indices in fidx])}

    @staticmethod
    def calc_source_strength_analytic2_fullfreq(sampler, freq_data,name):
        freqs = freq_data.fftfreq()
        strength = np.stack([freq_data.Q[i].diagonal() for i in range(freqs.shape[0])],axis=0)
        return {name: np.real(strength)}

    @staticmethod
    def calc_source_strength_analytic2_partfreq(sampler, freq_data, fidx, name):
        strength = np.array([freq_data.Q[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],dtype=complex)
        return {name: np.real(strength)}

    @staticmethod
    def calc_source_strength_analytic_custom_transfer_fullfreq(sampler, freq_data, ref_mic, name):
        freqs = freq_data.fftfreq()
        strength = np.real(np.stack([freq_data.Q[i].diagonal() for i in range(freqs.shape[0])],axis=0))
        transfer = freq_data.custom_transfer[:,ref_mic].copy()
        transfer *= transfer.conjugate()
        strength = strength*np.real(transfer)
        return {name: strength}

    @staticmethod
    def calc_source_strength_analytic_custom_transfer_partfreq(sampler, freq_data, fidx, ref_mic, name):
        full_strength = AnalyticSourceStrengthFeature.calc_source_strength_analytic_custom_transfer_fullfreq(sampler, freq_data, ref_mic, name)[name]
        strength = np.array([full_strength[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=float)
        return {name: strength}

    def get_feature_func(self):
        if isinstance(self.freq_data, PowerSpectraAnalytic):
            self.set_freq_limits()
            if self.fidx is None:
                if self.ref_mic is None:
                    return partial(self.calc_source_strength_analytic2_fullfreq, freq_data = self.freq_data, name=self.name)
                else:
                    return partial(self.calc_source_strength_analytic_custom_transfer_fullfreq, freq_data = self.freq_data, ref_mic=self.ref_mic, name=self.name)
            else:
                if self.ref_mic is None:
                    return partial(self.calc_source_strength_analytic2_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)
                else:
                    return partial(self.calc_source_strength_analytic_custom_transfer_partfreq, freq_data = self.freq_data,fidx = self.fidx, ref_mic=self.ref_mic, name=self.name)
        elif isinstance(self.freq_data, ac.BaseSpectra):
            if self.fidx is None:
                return partial(
                    self.calc_source_strength_analytic1_fullfreq, freq_data = self.freq_data, steer=self.steer, ref_mic=self.ref_mic, name=self.name)
            else:
                return partial(
                    self.calc_source_strength_analytic1_partfreq, freq_data = self.freq_data,fidx = self.fidx, steer=self.steer, ref_mic=self.ref_mic, name=self.name)
        else:
            raise NotImplementedError(f"No feature function with freq_data type {self.freq_data.__class__.__name__}.")


class EstimatedSourceStrengthFeature(SpectraFeature):

    name = Str("source_strength_estimated")
    ref_mic = Either(Int, None, default=None, desc="reference microphone index")

    @staticmethod
    def calc_source_strength_estimated1_fullfreq(sampler,freq_data, name):
        init_source = freq_data.source
        sources = get_point_sources_recursively(init_source)
        nfft = freq_data.fftfreq().shape[0]
        strength = np.zeros((nfft, len(sources)))
        for j, src in enumerate(sources):
            freq_data.source = src
            spectrogram = SpectrogramFeature.calc_spectrogram1(sampler,freq_data, name="spectrogram")["spectrogram"]
            strength[:,j] = np.real(np.real(spectrogram*spectrogram.conjugate())).mean(0).squeeze()
        freq_data.source = init_source # reset source in case of subsequent feature calculation
        return {name: strength}

    @staticmethod
    def calc_source_strength_estimated1_partfreq(sampler, freq_data, fidx, name):
        init_source = freq_data.source
        sources = get_point_sources_recursively(init_source)
        strength = np.zeros((len(fidx), len(sources)))
        for j, src in enumerate(sources):
            freq_data.source = src
            spectrogram = SpectrogramFeature.calc_spectrogram2(sampler,freq_data,fidx, name="spectrogram")["spectrogram"]
            strength[:,j] = np.real(np.real(spectrogram*spectrogram.conjugate())).mean(0).squeeze()
        freq_data.source = init_source # reset source in case of subsequent feature calculation
        return {name: strength}

    @staticmethod
    def calc_source_strength_estimated2_fullfreq(sampler, freq_data, name):
        freqs = freq_data.fftfreq()
        strength = np.stack([freq_data._Q[i].diagonal() for i in range(freqs.shape[0])],axis=0)
        return {name: np.real(strength)}

    @staticmethod
    def calc_source_strength_estimated2_partfreq(sampler, freq_data, fidx, name):
        strength = np.array([freq_data._Q[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],dtype=complex)
        return {name: np.real(strength)}

    @staticmethod
    def calc_source_strength_estimated3_fullfreq(sampler,freq_data, name):
        init_source = freq_data.source
        sources = get_point_sources_recursively(init_source)
        nfft = freq_data.fftfreq().shape[0]
        strength = np.zeros((nfft, len(sources)))
        for j, src in enumerate(sources):
            freq_data.source = src
            strength[:,j] = np.real(freq_data.csm[:,0,0])
        freq_data.source = init_source # reset source in case of subsequent feature calculation
        return {name: strength}

    @staticmethod
    def calc_source_strength_estimated3_partfreq(sampler,freq_data, fidx, name):
        init_source = freq_data.source
        sources = get_point_sources_recursively(init_source)
        strength = np.zeros((len(fidx), len(sources)))
        for j, src in enumerate(sources):
            freq_data.source = src
            csm = freq_data.csm[:]
            strength[:,j] = np.real(np.array(
                [csm[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],
                dtype=complex)).reshape((-1,))
        freq_data.source = init_source # reset source in case of subsequent feature calculation
        return {name: strength}

    @staticmethod
    def calc_source_strength_estimated_custom_transfer_fullfreq(sampler, freq_data, ref_mic, name):
        freqs = freq_data.fftfreq()
        strength = np.real(np.stack([freq_data._Q[i].diagonal() for i in range(freqs.shape[0])],axis=0))
        transfer = freq_data.custom_transfer[:,ref_mic].copy()
        transfer *= transfer.conjugate()
        strength = strength*np.real(transfer)
        return {name: strength}

    @staticmethod
    def calc_source_strength_estimated_custom_transfer_partfreq(sampler, freq_data, fidx, ref_mic, name):
        full_strength = EstimatedSourceStrengthFeature.calc_source_strength_estimated_custom_transfer_fullfreq(
            sampler, freq_data, ref_mic, name)[name]
        strength = np.array([full_strength[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=float)
        return {name: strength}

    def get_feature_func(self):

        if isinstance(self.freq_data, PowerSpectraAnalytic):
            self.set_freq_limits()
            if self.fidx is None:
                if self.ref_mic is None:
                    return partial(self.calc_source_strength_estimated2_fullfreq, freq_data = self.freq_data, name=self.name)
                else:
                    return partial(self.calc_source_strength_estimated_custom_transfer_fullfreq, freq_data = self.freq_data, ref_mic=self.ref_mic, name=self.name)
            else:
                if self.ref_mic is None:
                    return partial(self.calc_source_strength_estimated2_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)
                else:
                    return partial(self.calc_source_strength_estimated_custom_transfer_partfreq, freq_data = self.freq_data,fidx = self.fidx, ref_mic=self.ref_mic, name=self.name)

        elif isinstance(self.freq_data, ac.PowerSpectra):
            self.set_freq_limits()
            if self.fidx is None:
                return partial(self.calc_source_strength_estimated3_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_source_strength_estimated3_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)

        elif isinstance(self.freq_data, ac.RFFT):
            if self.fidx is None:
                return partial(self.calc_source_strength_estimated1_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_source_strength_estimated1_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)
        else:
            raise NotImplementedError(f"Unsupported freq_data type {self.freq_data.__class__}.")



class AnalyticNoiseStrengthFeature(SpectraFeature):

    name = Str("noise_strength_analytic")
    cross_strength = Enum(False) # can later be extended for cross-power values
    freq_data = Instance(ac.BaseSpectra, desc="cross spectral matrix calculation class")

    @staticmethod
    def calc_noise_strength_analytic1_fullfreq(sampler, freq_data, name):
        sources = get_uncorrelated_noise_source_recursively(freq_data.source)
        nfft = freq_data.fftfreq().shape[0]
        if len(sources) == 0:
            num_mics = get_point_sources_recursively(freq_data.source)[0].mics.num_mics
            return {name: np.zeros((freq_data.fftfreq().shape[0],num_mics))}
        elif len(sources) == 1:
            source = sources[0]
            mdim = source.mics.num_mics
            if isinstance(source.signal, ac.WNoiseGenerator):
                strength = np.ones((nfft,mdim))*(source.signal.rms**2/nfft)
            else:
                raise NotImplementedError(
                    f"Cannot handle source signal type {source.signal.__class__.__name__}.")
        else:
            raise ValueError("Only one uncorrelated noise source is supported.")
        return {name: strength}

    @staticmethod
    def calc_noise_strength_analytic1_partfreq(sampler, freq_data, fidx, name):
        sources = get_uncorrelated_noise_source_recursively(freq_data.source)
        if len(sources) == 0:
            num_mics = get_point_sources_recursively(freq_data.source)[0].mics.num_mics
            return {name: np.zeros((len(fidx),num_mics))}
        elif len(sources) == 1:
            nfft = freq_data.fftfreq().shape[0]
            source = sources[0]
            mdim = source.mics.num_mics
            if isinstance(source.signal, ac.WNoiseGenerator):
                strength = np.ones((nfft,mdim))*(source.signal.rms**2/nfft)
            else:
                raise NotImplementedError(
                    f"Cannot handle source signal type {source.signal.__class__.__name__}.")
        else:
            raise ValueError("Only one uncorrelated noise source is supported.")
        return {name: np.array([strength[indices[0]:indices[1]].sum(0) for indices in fidx])}

    @staticmethod
    def calc_noise_strength_analytic2_fullfreq(sampler, freq_data,name):
        if freq_data.noise is None:
            return {name: np.zeros((freq_data.fftfreq().shape[0],freq_data.steer.mics.num_mics))}
        else:
            freqs = freq_data.fftfreq()
            strength = np.stack([freq_data.noise[i].diagonal() for i in range(freqs.shape[0])],axis=0)
            return {name: np.real(strength)}

    @staticmethod
    def calc_noise_strength_analytic2_partfreq(sampler, freq_data, fidx, name):
        if freq_data.noise is None:
            return {name: np.zeros((len(fidx),freq_data.steer.mics.num_mics))}
        else:
            strength = np.array([freq_data.noise[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],dtype=complex)
            return {name: np.real(strength)}

    def get_feature_func(self):
        if isinstance(self.freq_data, PowerSpectraAnalytic):
            self.set_freq_limits()
            if self.fidx is None:
                return partial(self.calc_noise_strength_analytic2_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_noise_strength_analytic2_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)

        elif isinstance(self.freq_data, ac.BaseSpectra):
            if self.fidx is None:
                return partial(self.calc_noise_strength_analytic1_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_noise_strength_analytic1_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)
        else:
            raise NotImplementedError(f"No feature function with freq_data type {self.freq_data.__class__.__name__}.")



class EstimatedNoiseStrengthFeature(SpectraFeature):

    name = Str("noise_strength_estimated")

    @staticmethod
    def calc_noise_strength_estimated1_fullfreq(sampler,freq_data, name):
        sources = get_uncorrelated_noise_source_recursively(freq_data.source)
        if len(sources) == 0:
            num_mics = get_point_sources_recursively(freq_data.source)[0].mics.num_mics
            return {name: np.zeros((freq_data.fftfreq().shape[0],num_mics))}
        elif len(sources) == 1:
            freq_data.source = sources[0]
            spectrogram = SpectrogramFeature.calc_spectrogram1(sampler,freq_data, name="spectrogram")["spectrogram"]
            strength = np.real(np.real(spectrogram*spectrogram.conjugate())).mean(0)
            return {name: strength}
        else:
            raise ValueError("Only one uncorrelated noise source is supported.")

    @staticmethod
    def calc_noise_strength_estimated1_partfreq(sampler, freq_data, fidx, name):
        sources = get_uncorrelated_noise_source_recursively(freq_data.source)
        freq_data.fftfreq().shape[0]
        if len(sources) == 0:
            num_mics = get_point_sources_recursively(freq_data.source)[0].mics.num_mics
            return {name: np.zeros((len(fidx),num_mics))}
        elif len(sources) == 1:
            freq_data.source = sources[0]
            spectrogram = SpectrogramFeature.calc_spectrogram2(sampler,freq_data,fidx, name="spectrogram")["spectrogram"]
            strength = np.real(np.real(spectrogram*spectrogram.conjugate())).mean(0)
            return {name: strength}
        else:
            raise ValueError("Only one uncorrelated noise source is supported.")

    @staticmethod
    def calc_noise_strength_estimated2_fullfreq(sampler, freq_data, name):
        freqs = freq_data.fftfreq()
        if freq_data._noise is None:
            return {name: np.zeros((freqs.shape[0],freq_data.steer.mics.num_mics))}
        strength = np.stack([freq_data._noise[i].diagonal() for i in range(freqs.shape[0])],axis=0)
        return {name: np.real(strength)}

    @staticmethod
    def calc_noise_strength_estimated2_partfreq(sampler, freq_data, fidx, name):
        if freq_data._noise is None:
            return {name: np.zeros((len(fidx),freq_data.steer.mics.num_mics))}
        strength = np.array([freq_data._noise[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],dtype=complex)
        return {name: np.real(strength)}

    def get_feature_func(self):
        if isinstance(self.freq_data, PowerSpectraAnalytic):
            self.set_freq_limits()
            if self.fidx is None:
                return partial(self.calc_noise_strength_estimated2_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_noise_strength_estimated2_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)

        elif isinstance(self.freq_data, ac.RFFT):
            if self.fidx is None:
                return partial(self.calc_noise_strength_estimated1_fullfreq, freq_data = self.freq_data, name=self.name)
            else:
                return partial(self.calc_noise_strength_estimated1_partfreq, freq_data = self.freq_data,fidx = self.fidx, name=self.name)
        else:
            raise NotImplementedError(f"Unsupported freq_data type {self.freq_data.__class__}.")


class BaseFeatureCollection(HasPrivateTraits):
    """
    BaseFeatureCollection base class for handling feature funcs.

    Attributes
    ----------
    feature_funcs : list
        List of feature_funcs.
    """

    feature_funcs = List(desc="list of feature_funcs")
    feature_tf_encoder_mapper = Dict(desc="feature encoder mapper")
    feature_tf_shape_mapper = Dict(desc="feature shape mapper")
    feature_tf_dtype_mapper = Dict(desc="feature dtype mapper")

    def __init__(self):
        HasPrivateTraits.__init__(self)
        if TF_FLAG:
            from acoupipe.writer import int64_feature, int_list_feature
            self.feature_tf_encoder_mapper = {"idx" : int64_feature,
                                            "seeds" : int_list_feature,}
            self.feature_tf_shape_mapper = {"idx" : (),
                                            "seeds" : (None,2)}
            self.feature_tf_dtype_mapper = {"idx" : "int64",
                                            "seeds" : "int64"}


    def add_feature_func(self, feature_func):
        """
        Add a feature_func to the BaseFeatureCollection.

        Parameters
        ----------
        feature_func : str
            Feature to be added.
        """
        self.feature_funcs.append(feature_func)

    def get_feature_funcs(self):
        """
        Get all feature_funcs of the BaseFeatureCollection.

        Returns
        -------
        list
            List of feature_funcs.
        """
        def calc_features(sampler, feature_funcs):
            data = {}
            for ffunc in feature_funcs:
                data.update(ffunc(sampler=sampler))
            return data
        return partial(calc_features, feature_funcs=self.feature_funcs)


class BaseFeatureCollectionBuilder(HasPrivateTraits):
    """
    BaseFeatureCollectionBuilder base class for building a BaseFeatureCollection.

    Attributes
    ----------
    feature_collection : BaseFeatureCollection
        BaseFeatureCollection object.
    """

    feature_collection = Instance(BaseFeatureCollection, desc="BaseFeatureCollection object")

    def add_custom(self, feature_func):
        """
        Add a custom feature to the BaseFeatureCollection.

        Parameters
        ----------
        feature_func : str
            Feature to be added.
        """
        self.feature_collection.add_feature_func(feature_func)


    def build(self):
        """
        Build a BaseFeatureCollection.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        return self.feature_collection
