from functools import partial
from warnings import warn

import acoular as ac
import numpy as np
from numpy import array, imag, newaxis, real, triu_indices
from numpy.linalg import eigh
from traits.api import Callable, Either, Enum, Float, HasPrivateTraits, Instance, Int, List, Property, Str, Tuple

from acoupipe.config import TF_FLAG
from acoupipe.datasets.precision import (
    NUMPY_COMPLEX_DTYPES,
    NUMPY_FLOAT_DTYPES,
    TF_COMPLEX_DTYPES,
    TF_FLOAT_DTYPES,
    TF_INT_DTYPES,
)
from acoupipe.datasets.precision import precision as PRECISION_CONFIG
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic, TransferBase
from acoupipe.datasets.utils import (
    get_frequency_index_range,
    get_point_sources_recursively,
    get_uncorrelated_noise_source_recursively,
)

if TF_FLAG:
    from acoupipe.writer import complex_list_feature, float_list_feature, int64_feature
else:
    float_list_feature = None

class Feature(HasPrivateTraits):
    """Feature base class for handling feature funcs.

    Attributes
    ----------
    name : str
        Name of the feature.
    shape : tuple
        Shape of the feature.
    """

    name = Str
    shape = Tuple
    feature_func = Property()
    _feature_func = Callable

    def get_tf_shape_mapper(self):
        return {self.name : self.shape}

    def get_tf_dtype_mapper(self):
        return {self.name : self.tf_dtype}

    def _get_feature_func(self):
        """Will return a method depending on the class parameters."""
        return partial(
            self._feature_func, name=self.name, dtype=self.dtype)

    def _set_feature_func(self, func):
        self._feature_func = func


class IntFeature(Feature):
    """Feature base class for handling integer type features.

    Attributes
    ----------
    name : str
        Name of the feature.
    shape : tuple
        Shape of the feature.
    dtype : str
        The numpy data type of the feature.
    tf_dtype : str
        The tensorflow data type of the feature.
    """

    dtype = Property(desc="numpy data type")
    tf_dtype = Property(desc="tensorflow data type")
    _dtype = TF_INT_DTYPES
    _tf_dtype = TF_FLOAT_DTYPES

    def get_tf_encoder_mapper(self):
        return {self.name : int64_feature}

    def get_tf_shape_mapper(self):
        return {self.name : self.shape}

    def get_tf_dtype_mapper(self):
        return {self.name : self.tf_dtype}

    def _get_dtype(self):
        if PRECISION_CONFIG.int is None:
            return self._dtype
        else:
            return PRECISION_CONFIG.int

    def _set_dtype(self, value):
        if PRECISION_CONFIG.int is not None \
            and PRECISION_CONFIG.int != value:
            warn(
                (f"Setting the dtype of {self.__class__} will have no effect! "
                f"The dtype is superseeded by the global precision setting {PRECISION_CONFIG.int}."),
                UserWarning,
                stacklevel=2)
        self._dtype = value

    def _get_tf_dtype(self):
        if PRECISION_CONFIG.tf_int is None:
            return self._tf_dtype
        else:
            return PRECISION_CONFIG.tf_int

    def _set_tf_dtype(self, value):
        if PRECISION_CONFIG.tf_int is not None \
            and PRECISION_CONFIG.tf_int != value:
            warn(
                (f"Setting the tf_dtype of {self.__class__} will have no effect! "
                f"The tf_dtype is superseeded by the global precision setting {PRECISION_CONFIG.tf_int}."),
                UserWarning,
                stacklevel=2)
        self._tf_dtype = value


class FloatFeature(Feature):
    """Feature base class for handling float type features.

    Attributes
    ----------
    name : str
        Name of the feature.
    shape : tuple
        Shape of the feature.
    dtype : str
        The numpy data type of the feature.
    tf_dtype : str
        The tensorflow data type of the feature.
    """

    dtype = Property(desc="numpy data type")
    tf_dtype = Property(desc="tensorflow data type")
    _dtype = NUMPY_FLOAT_DTYPES
    _tf_dtype = TF_FLOAT_DTYPES

    def get_tf_encoder_mapper(self):
        return {self.name : float_list_feature}

    def get_tf_shape_mapper(self):
        return {self.name : self.shape}

    def get_tf_dtype_mapper(self):
        return {self.name : self.tf_dtype}

    def _get_dtype(self):
        if PRECISION_CONFIG.float is None:
            return self._dtype
        else:
            return PRECISION_CONFIG.float

    def _set_dtype(self, value):
        if PRECISION_CONFIG.float is not None \
            and PRECISION_CONFIG.float != value:
            warn(
                (f"Setting the dtype of {self.__class__} will have no effect! "
                f"The dtype is superseeded by the global precision setting {PRECISION_CONFIG.float}."),
                UserWarning,
                stacklevel=2)
        self._dtype = value

    def _get_tf_dtype(self):
        if PRECISION_CONFIG.tf_float is None:
            return self._tf_dtype
        else:
            return PRECISION_CONFIG.tf_float

    def _set_tf_dtype(self, value):
        if PRECISION_CONFIG.tf_float is not None \
            and PRECISION_CONFIG.tf_float != value:
            warn(
                (f"Setting the tf_dtype of {self.__class__} will have no effect! "
                f"The tf_dtype is superseeded by the global precision setting {PRECISION_CONFIG.tf_float}."),
                UserWarning,
                stacklevel=2)
        self._tf_dtype = value


class ComplexFeature(Feature):
    """Feature base class for handling complex-valued features.

    Attributes
    ----------
    name : str
        Name of the feature.
    shape : tuple
        Shape of the feature.
    dtype : str
        The numpy data type of the feature.
    tf_dtype : str
        The tensorflow data type of the feature.
    """

    dtype = Property(desc="numpy data type")
    tf_dtype = Property(desc="tensorflow data type")
    _dtype = NUMPY_COMPLEX_DTYPES
    _tf_dtype = TF_COMPLEX_DTYPES

    def get_tf_encoder_mapper(self):
        return {self.name : complex_list_feature}

    def _get_dtype(self):
        if PRECISION_CONFIG.complex is None:
            return self._dtype
        else:
            return PRECISION_CONFIG.complex

    def _set_dtype(self, value):
        if PRECISION_CONFIG.complex is not None \
            and PRECISION_CONFIG.complex != value:
            warn(
                (f"Setting the dtype of {self.__class__} will have no effect! "
                f"The dtype is superseeded by the global precision setting {PRECISION_CONFIG.complex}."),
                UserWarning,
                stacklevel=2)
        self._dtype = value

    def _get_tf_dtype(self):
        if PRECISION_CONFIG.tf_complex is None:
            return self._tf_dtype
        else:
            return PRECISION_CONFIG.tf_complex

    def _set_tf_dtype(self, value):
        if PRECISION_CONFIG.tf_complex is not None \
            and PRECISION_CONFIG.tf_complex != value:
            warn(
                (f"Setting the tf_dtype of {self.__class__} will have no effect! "
                f"The tf_dtype is superseeded by the global precision setting {PRECISION_CONFIG.tf_complex}."),
                UserWarning,
                stacklevel=2)
        self._tf_dtype = value


class LocFeature(FloatFeature):

    name = Str("loc")
    freq_data = Instance(ac.BaseSpectra, desc="cross spectral matrix calculation class")
    shape = Either((3, None), Tuple)

    @staticmethod
    def calc_loc1(sampler, freq_data, dtype, name):
        sources = get_point_sources_recursively(freq_data.source)
        locs = np.array([src.loc for src in sources]).T
        return {name: locs.astype(dtype)}

    @staticmethod
    def calc_loc2(sampler, freq_data, dtype, name):
        if isinstance(freq_data.transfer.grid, ac.ImportGrid):
            return {name: freq_data.transfer.grid.gpos_file.astype(dtype)}
        else:
            return {name: freq_data.transfer.grid.gpos.astype(dtype)}

    def _get_feature_func(self):
        if isinstance(self.freq_data, PowerSpectraAnalytic):
            return partial(self.calc_loc2, freq_data = self.freq_data, dtype=self.dtype, name=self.name)
        elif isinstance(self.freq_data, ac.BaseSpectra):
            return partial(self.calc_loc1, freq_data = self.freq_data, dtype=self.dtype, name=self.name)
        else:
            raise NotImplementedError(f"Unknown freq_data type {self.freq_data.__class__.__name__}.")


class TimeDataFeature(FloatFeature):
    """TimeDataFeature class for handling time data.

    Attributes
    ----------
    name : str
        Name of the feature (default='time_data').
    time_data : instance of class acoular.SamplesGenerator
        The source delivering the time data.
    dtype : str
        The numpy data type of the feature. (default='float64')
    tf_dtype : str
        The tensorflow data type of the feature. (default='float32')

    """

    name = Str("time_data")
    shape = Tuple(default=(None, None))
    time_data = Either(
        Instance(ac.SignalGenerator),
        Instance(ac.SamplesGenerator),
        List(Instance(ac.SamplesGenerator)),
        List(Instance(ac.SignalGenerator)),
        )

    def _get_feature_func(self):
        """Return the callable for calculating the time data."""
        def _calc_time_data(sampler, time_data, name, dtype):
            if isinstance(time_data, list):
                if isinstance(time_data[0], ac.SamplesGenerator):
                    time_data = ac.SourceMixer(sources=time_data)
                    return {
                        name: ac.tools.return_result(time_data).astype(dtype)
                        }
                elif isinstance(time_data[0], ac.SignalGenerator):
                    time_data = np.concatenate([_.signal()[:,np.newaxis] for _ in time_data], axis=1)
                    return {
                        name: time_data.astype(dtype)
                        }
            elif isinstance(time_data, ac.SignalGenerator):
                return {name: time_data.signal()[:,np.newaxis].astype(dtype)}
            else:
                return {
                    name: ac.tools.return_result(time_data).astype(dtype)
                    }
        return partial(_calc_time_data,time_data=self.time_data, name=self.name, dtype=self.dtype)


class SourcemapFeature(FloatFeature):
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
    """

    name = Str("sourcemap")
    beamformer = Instance(ac.BeamformerBase, desc="beamformer")
    f = Either(None, Float, List(Float), desc="frequency")
    num = Int
    shape = Either((None,None,None), Tuple)

    @staticmethod
    def calc_beamformer1(sampler, beamformer, f, num, dtype, name):
        sm = array([beamformer.synthetic(freq,num=num) for freq in f], dtype=dtype)
        return {name: sm}

    @staticmethod
    def calc_beamformer2(sampler, beamformer, dtype, name):
        f = beamformer.freq_data.fftfreq()
        sm = array([beamformer.synthetic(freq,num=0) for freq in f], dtype=dtype)
        return {name: sm}

    def _get_feature_func(self):
        """Return the callable for calculating the sourcemap."""
        if self.f is not None:
            if isinstance(self.f, (float, int)):
                f = [self.f]
            else:
                f = self.f
            return partial(
                self.calc_beamformer1, beamformer=self.beamformer, f=f, num=self.num, dtype=self.dtype, name=self.name)
        else:
            return partial(
                self.calc_beamformer2, beamformer=self.beamformer,  dtype=self.dtype, name=self.name)



class TargetmapFeature(FloatFeature):

    name = Str("targetmap")
    shape = Either((None,None,None), Tuple)
    loc_callable = Callable
    strength_callable = Callable
    grid = Instance(ac.Grid)

    @staticmethod
    def get_targetmap(sampler, grid, loc_callable, strength_callable, dtype, name):
        loc = list(loc_callable(sampler=sampler).values())[0]
        strength = list(strength_callable(sampler=sampler).values())[0]
        # create target map
        if type(grid) is ac.RectGrid:
            loc = loc[:2]
        elif type(grid) is ac.RectGrid3D:
            loc = loc[:3]
        else:
            raise NotImplementedError(f"Unknown grid type {type(grid)}.")
            # other grid types in Acoular do not provide an index method!
        target_map = np.zeros((strength.shape[0],)+grid.shape, dtype=dtype)
        for j in range(loc.shape[1]):
            index = grid.index(*loc[:, j])
            target_map[(slice(None),) + index] += strength[:,j]
        return {name: target_map}

    def _get_feature_func(self):
        return partial(
            self.get_targetmap, grid=self.grid,
            loc_callable=self.loc_callable, strength_callable=self.strength_callable,
            dtype=self.dtype, name=self.name)



class SpectraFeature(ComplexFeature):
    """Handles the calculation of features in the frequency domain.

    Attributes
    ----------
    name : str
        Name of the feature.
    freq_data : instance of class acoular.BaseSpectra
        The frequency data to calculate the feature for.
    f : float
        the frequency (or center frequencies) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)
    """

    freq_data = Instance(ac.BaseSpectra, desc="spectrogram")
    f = Either(None, Float, List(Float), desc="frequency")
    num = Int

    @staticmethod
    def get_spectral_coeff(arr, dtype, fidx, sum_axis=0):
        return np.array([arr[indices[0]:indices[1]].sum(sum_axis) for indices in fidx],dtype=dtype)

    def _get_fidx(self):
        """Return list of tuples containing the start and end indices of the frequency bands to be considered."""
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


class SpectrogramFeature(SpectraFeature):
    """SpectrogramFeature class for handling spectrogram features.

    Attributes
    ----------
    name : str
        Name of the feature (default='spectrogram').
    freq_data : instance of class acoular.FFTSpectra
        The object which calculates the spectrogram data.
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)
    fidx : list of tuples
        List of tuples containing the start and end indices of the frequency bands to be considered.
    """

    name = Str("spectrogram")
    freq_data = Instance(ac.FFTSpectra, desc="spectrogram")
    shape = Either((None,None,None), Tuple)

    @staticmethod
    def calc_spectrogram(sampler, freq_data, fidx, dtype, name):
        spectrogram = np.stack(
            list((_.copy() for _ in freq_data.result())),dtype=dtype, axis=0)
        if fidx is not None:
            spectrogram = np.array(
                [spectrogram[:,indices[0]:indices[1]].sum(1) for indices in fidx],
                    dtype=dtype).swapaxes(0,1)
        return {name:spectrogram}

    def _get_feature_func(self):
        fidx = self._get_fidx()
        return partial(
            self.calc_spectrogram, freq_data = self.freq_data, dtype=self.dtype, name=self.name, fidx = fidx)


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
    shape = Either((None,None,None), Tuple)

    @staticmethod
    def calc_csm(sampler, freq_data, fidx, dtype, name):
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
        if fidx is not None:
            csm = SpectraFeature.get_spectral_coeff(csm, dtype, fidx)
        return {name :csm.astype(dtype)}

    def _get_feature_func(self):
        """Return the callable for calculating the cross-spectral matrix."""
        fidx = self._get_fidx()
        return partial(
                self.calc_csm, freq_data = self.freq_data, dtype=self.dtype, name=self.name, fidx = fidx)


class CSMtriuFeature(FloatFeature, CSMFeature):

    name = Str("csmtriu")
    freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix")
    shape = Either((None,None,None), Tuple)

    @staticmethod
    def transform(csm, dtype):
        csmtriu_real = np.zeros(csm.shape, dtype=dtype)
        csm_triu_imag = np.zeros(csm.shape, dtype=dtype)
        num_mics = csm.shape[1]
        for i in range(csm.shape[0]):
            csmtriu_real[i][triu_indices(num_mics)] = real(csm[i])[triu_indices(num_mics)] # add real part at upper triangular matrix
            csm_triu_imag[i][triu_indices(num_mics)] = imag(csm[i])[triu_indices(num_mics)]
        return csmtriu_real + csm_triu_imag.transpose(0,2,1)

    @staticmethod
    def calc_csmtriu(sampler, freq_data, dtype, fidx, name):
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
        if fidx is not None:
            csm = SpectraFeature.get_spectral_coeff(csm, csm.dtype, fidx)
        return {name: CSMtriuFeature.transform(csm, dtype)}

    def _get_feature_func(self):
        fidx = self._get_fidx()
        return partial(
            self.calc_csmtriu, freq_data = self.freq_data, dtype=self.dtype, name=self.name, fidx = fidx)


class EigmodeFeature(SpectraFeature):

        name = Str("eigmode")
        freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix calculation class")
        shape = Either((None,None,None), Tuple)

        @staticmethod
        def transform(csm):
            eva, eve = eigh(csm)
            return eva[:,newaxis,:]*eve[:]

        @staticmethod
        def calc_eigmode(sampler, freq_data, fidx, dtype, name):
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
            csm = freq_data.csm.astype(dtype)
            if fidx is not None:
                csm = SpectraFeature.get_spectral_coeff(csm, dtype, fidx)
            return {name: EigmodeFeature.transform(csm)}


        def _get_feature_func(self):
            fidx = self._get_fidx()
            return partial(
                self.calc_eigmode, freq_data = self.freq_data, dtype=self.dtype, name=self.name, fidx = fidx)


class CSMAnalytic(CSMFeature):

    name = Str("csm")
    freq_data = Instance(PowerSpectraAnalytic, desc="cross spectral matrix calculation class")
    shape = Either((None,None,None), Tuple)
    mode = Enum(("analytic","wishart"))
    csm_type = Enum(("csm", "noise", "source"))

    @staticmethod
    def get_csm(mode, csm_type, freq_data, dtype):
        if mode == "analytic":
            if csm_type == "csm" and freq_data.mode == "analytic":
                csm = freq_data.csm
            elif csm_type == "noise":
                csm = freq_data.noise
            elif csm_type == "source":
                csm = freq_data.Q
            elif csm_type == "csm" and freq_data.mode == "wishart":
                raise ValueError("Cannot calculate analytic csm from wishart mode.")
            else:
                raise ValueError(f"Unknown csm_type {csm_type}.")
        else:
            if csm_type == "csm" and freq_data.mode == "wishart":
                csm = freq_data.csm
            elif csm_type == "noise":
                csm = freq_data._noise
            elif csm_type == "source":
                csm = freq_data._Q
            elif csm_type == "csm" and freq_data.mode == "analytic":
                raise ValueError("Cannot calculate wishart csm from analytic mode.")
            else:
                raise ValueError(f"Unknown csm_type {csm_type}.")
        if csm is None:
            return None
        return csm.astype(dtype)

    @staticmethod
    def calc_csm(sampler, csm_type, mode, freq_data, fidx, dtype, name):
        csm = CSMAnalytic.get_csm(mode, csm_type, freq_data, dtype)
        if fidx is None:
            nf = freq_data.fftfreq().shape[0]
        else:
            nf = len(fidx)

        if csm is None:
            return {name: np.zeros((nf,freq_data.transfer.mics.num_mics,freq_data.transfer.mics.num_mics), dtype=dtype)}
        elif fidx is None:
            return {name: csm}
        else:
            return {name: SpectraFeature.get_spectral_coeff(csm, dtype, fidx)}

    def _get_feature_func(self):
        fidx = self._get_fidx()
        return partial(
            self.calc_csm, csm_type=self.csm_type, mode=self.mode, freq_data = self.freq_data, fidx=fidx,
            dtype=self.dtype, name=self.name)


class CSMDiagonalAnalytic(FloatFeature, CSMAnalytic):

    name = Str("noise_strength")
    freq_data = Instance(PowerSpectraAnalytic, desc="cross spectral matrix calculation class")
    shape = Either((None,None), Tuple)

    @staticmethod
    def calc_csm(sampler, csm_type, mode, freq_data, fidx, dtype, name):
        csm = CSMAnalytic.get_csm(mode, csm_type, freq_data, dtype)

        if fidx is None:
            nf = freq_data.fftfreq().shape[0]
        else:
            nf = len(fidx)

        if csm is None:
            return {name: np.zeros((nf,freq_data.transfer.mics.num_mics), dtype=dtype)}
        elif fidx is None:
            return {name: np.stack([csm[i].diagonal() for i in range(nf)],axis=0, dtype=dtype)}
        else:
            return {name: np.array([csm[indices[0]:indices[1]].sum(0).diagonal() for indices in fidx],dtype=dtype)}

    def _get_feature_func(self):
        fidx = self._get_fidx()
        return partial(
            self.calc_csm, csm_type=self.csm_type, mode=self.mode, freq_data = self.freq_data, fidx=fidx,
            dtype=self.dtype, name=self.name)


class CSMDiagonalWelch(FloatFeature, CSMFeature):

    name = Str("csm")
    freq_data = Instance(ac.PowerSpectra, desc="cross spectral matrix calculation class")
    transfer = Instance(TransferBase, desc="transfer function")
    shape = Either((None,None), Tuple)
    mode = Enum(("analytic","welch"))
    csm_type = Enum(("noise", "source"))

    @staticmethod
    def calc_noise(sampler, mode, freq_data, fidx, dtype, name):
        sources = get_uncorrelated_noise_source_recursively(freq_data.source)
        nfft = freq_data.fftfreq().shape[0]
        if len(sources) == 0:
            num_mics = get_point_sources_recursively(freq_data.source)[0].mics.num_mics
            return {name: np.zeros((freq_data.fftfreq().shape[0],num_mics))}
        elif len(sources) == 1:
            source = sources[0]
            if not isinstance(source.signal, ac.WNoiseGenerator):
                raise NotImplementedError(
                    f"Cannot handle source signal type {source.signal.__class__.__name__}.")
            mdim = source.mics.num_mics
            if mode == "analytic":
                strength = np.ones((nfft,mdim), dtype=dtype)*(source.signal.rms**2/nfft)
                if fidx is not None:
                    strength = SpectraFeature.get_spectral_coeff(strength, dtype, fidx)
            else:
                fft_spectra = ac.FFTSpectra(
                    window=freq_data.window, block_size=freq_data.block_size,
                    overlap=freq_data.overlap, source=source)
                spectrogram = SpectrogramFeature.calc_spectrogram(
                    sampler,fft_spectra, fidx=fidx, dtype="complex128", name="s")["s"]
                strength = np.real(spectrogram*spectrogram.conjugate()).mean(0).astype(dtype)
        else:
            raise ValueError("Only one uncorrelated noise source is supported.")
        return {name: strength}

    @staticmethod
    def calc_source(sampler, mode, freq_data, transfer, fidx, dtype, name):
        init_source = freq_data.source
        sources = get_point_sources_recursively(freq_data.source)
        if mode == "analytic":
            nfft = freq_data.fftfreq().shape[0]
            strength = np.zeros((nfft, len(sources)), dtype=dtype)
            transfer.grid = ac.ImportGrid()
            for j, source in enumerate(sources):
                # H = np.zeros((freqs.shape[0], len(sources)),dtype=complex)
                # for i, f in enumerate(freqs):
                #     H[i] = transfer.transfer(f)[:,0]
                # H *= H.conjugate()
                # p = np.zeros((len(sources),))

                # if isinstance(source, ac.PointSourceConvolve):
                #     ir = source.kernel[:,ref_mic].copy()
                #     tf = blockwise_transfer(ir[np.newaxis], blocksize=freq_data.block_size).squeeze()
                #     strength[:,j] = np.real(tf*tf.conjugate())*source.signal.rms**2/nfft
                if isinstance(source, ac.PointSource):
                    transfer.grid = ac.ImportGrid(gpos_file=np.array(source.loc)[:,np.newaxis])
                    if isinstance(source.signal, ac.WNoiseGenerator):
                        strength[:,j] = np.ones(nfft)*(source.signal.rms/transfer.r0)**2/nfft
                    else:
                        raise NotImplementedError(
                            f"Cannot handle source signal type {source.signal.__class__.__name__}.")
                else:
                    raise NotImplementedError(
                        f"Cannot handle source type {source.__class__.__name__}.")
            if fidx is not None:
                strength = SpectraFeature.get_spectral_coeff(strength, dtype, fidx)
        else:
            if fidx is None:
                nfft = freq_data.fftfreq().shape[0]
            else:
                nfft = len(fidx)
            strength = np.zeros((nfft, len(sources)), dtype=dtype)
            for j, src in enumerate(sources):
                freq_data.source = src
                res = freq_data.csm.astype(dtype).reshape((-1))
                if fidx is not None:
                    res = SpectraFeature.get_spectral_coeff(res, dtype, fidx)
                strength[:,j] = res
            freq_data.source = init_source # reset source in case of subsequent feature calculation
        return {name: strength}

    def _get_feature_func(self):
        fidx = self._get_fidx()
        if self.csm_type == "noise":
            return partial(
                    self.calc_noise, mode=self.mode, freq_data = self.freq_data, fidx = fidx,
                    dtype=self.dtype, name=self.name)
        else:
            return partial(
                    self.calc_source, transfer=self.transfer, mode=self.mode, freq_data = self.freq_data, fidx = fidx,
                    dtype=self.dtype, name=self.name)

