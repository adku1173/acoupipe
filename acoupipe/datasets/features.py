from acoular import config
from acoupipe import float_list_feature, PlanarSourceMapEvaluator 
from .helper import get_frequency_index_range
from numpy import zeros, array, float32, concatenate, real, imag, triu_indices, newaxis, transpose,\
    conj, concatenate
import numba
import warnings
#from .sbl import SBL, Options
warnings.filterwarnings("ignore") # suppress pickling warnings


def get_eigmap(beamformer, n=16, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculates the sourcemap with a specified beamformer instance
    of class (or derived class) of type acoular.BeamformerBase.

    Parameters
    ----------
    beamformer : instance of class acoular.BeamformerBase
        beamformer to calculate the source map feature
    n : int
        eigenvalue components to calculate (n strongest)
    f : float
        frequency to evaluate
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
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        sourcemap feature of either shape (n,nxsteps,nysteps) or (n,B/2+1,n,nxsteps,nysteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if not f:
        eig_maps = []
        for i in range(n):
            beamformer.n = (-1)-i # get the sourcemap for the n largest eigenvalues
            eig_maps.append([beamformer.synthetic(f,num=num) for f in beamformer.freq_data.fftfreq()])
        return array(eig_maps)
    else:
        eig_maps = []
        for i in range(n):
            beamformer.n = (-1)-i # get the sourcemap for the n largest eigenvalues
            eig_maps.append(beamformer.synthetic(f,num=num))
        return array(eig_maps) # sort from largest to smallest 



def get_sourcemap(beamformer, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculates the sourcemap with a specified beamformer instance
    of class (or derived class) of type acoular.BeamformerBase.

    Parameters
    ----------
    beamformer : instance of class acoular.BeamformerBase
        beamformer to calculate the source map feature
    f : float
        frequency to evaluate
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
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        sourcemap feature of either shape (1,nxsteps,nysteps,nzsteps) or (B/2+1,nxsteps,nysteps,nzsteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if not f:
        sm = array([beamformer.synthetic(f,num=num) for f in beamformer.freq_data.fftfreq()])
    else:
        sm = array([beamformer.synthetic(freq,num=num) for freq in f]) 
    if sm.ndim == 3:
        sm = sm[...,newaxis]
    return sm

# def get_SBLmap(spectra_inout, steer, fidx=None):
#     """
#     """
#     options = Options(convergence_error=10 ** (-8), gamma_range=10 ** (-4), convergence_maxiter=5000, convergence_min_iteration=1, status_report=1, fixedpoint=1, Nsource=len(spectra_inout.source.sources), flag=0)    
#     fftfreq = spectra_inout.fftfreq()
#     A = zeros((steer.mics.num_mics,steer.grid.size,fftfreq.shape[0]), dtype = complex)
#     for v in range(fftfreq.shape[0]):
#         A[:,:,v] = steer.transfer(fftfreq[v]).T
#     if not fidx:
#         NotImplementedError("full frequency support currently not implemented")
#     else:
#         source_maps = []        
#         for freq_index in fidx:
#             nfreq = freq_index[1]-freq_index[0]
#             Ysignal = transpose(array(list(spectra_inout.result())),[2,0,1])
#             gamma, report = SBL(A[:,:,freq_index[0]:freq_index[1]], Ysignal[:,:,freq_index[0]:freq_index[1]], options)
#             gamma *= nfreq
#             source_maps.append(gamma)
#         return array(source_maps) 

def get_csm(power_spectra, fidx=None, cache_dir=None, num_threads=1):
    """Calculates the cross-spectral matrix (CSM). 

    Parameters
    ----------
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the CSM
    fidx : int, optional
        frequency index at which the CSM is returned, by default None, meaning that the
        CSM for all frequency coefficients will be returned
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        The cross-spectral matrix with shape (1,M,M,2) or (B/2+1,M,M,2) if fidx=None. 
        B: Blocksize of the FFT. M: Number of microphones.Real values will 
        be stored at the first entry of the first dimension. 
        Imaginary values are stored at the second entry of the last dimension.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    csm = power_spectra.csm[:]
    if fidx:
        csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)
    return concatenate([real(csm)[...,newaxis], imag(csm)[...,newaxis]],axis=3, dtype=float32)


def _transform_csm(csm):
    csm_list = []
    num_mics = csm.shape[1]
    for i in range(csm.shape[0]):
        csm_recover_real = zeros((num_mics,num_mics),dtype=float32)
        csm_recover_imag = zeros((num_mics,num_mics),dtype=float32)
        csm_recover_real[triu_indices(num_mics)] = real(csm[i])[triu_indices(num_mics)] # add real part at upper triangular matrix
        csm_recover_imag[triu_indices(num_mics)] = imag(csm[i])[triu_indices(num_mics)]
        csm_list.append(csm_recover_real + csm_recover_imag.T)
    return array(csm_list,dtype=float32)[...,newaxis]

def get_csmtriu(power_spectra, fidx=None, cache_dir=None, num_threads=1): 
    """Calculates the cross-spectral matrix (CSM) and returns it with the same representation
    as in:

    Paolo Castellini, Nicola Giulietti, Nicola Falcionelli, Aldo Franco Dragoni, Paolo Chiariotti,
    A neural network based microphone array approach to grid-less noise source localization,
    Applied Acoustics, Volume 177, 2021.

    Parameters
    ----------
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the CSM
    fidx : int, optional
        frequency index at which the CSM is returned, by default None, meaning that the
        CSM for all frequency coefficients will be returned
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        The cross-spectral matrix with shape (1,M,M,1) or (B/2+1,M,M,1) if fidx=None. 
        B: Blocksize of the FFT. M: Number of microphones.Real values will 
        be stored at the upper triangular matrix. 
        Imaginary values will be stored at the lower triangular matrix.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if not fidx:
        return _transform_csm(power_spectra.csm)    
    else:
        csm = _transform_csm(power_spectra.csm)  
        return array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=float32)

def get_spectrogram(spectra_inout, fidx=None, cache_dir=None, num_threads=1):
    """Calculates the cross-spectral matrix (CSM). 

    Parameters
    ----------
    spectra_inout : instance of spectacoular.SpectraInOut
        object to calculate the CSM
    fidx : int, optional
        frequency index at which the CSM is returned, by default None, meaning that the
        CSM for all frequency coefficients will be returned
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        The complex spectrogram matrix with shape (2,B/2+1,T,M) if fidx=None. 
        B: Blocksize of the FFT. M: Number of microphones.Real values will 
        be stored at the first entry of the first dimension. 
        Imaginary values will be stored at the second entry of the first dimension.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    res = array(list(spectra_inout.result())).swapaxes(0,1)        
    if fidx: # list with freq index tuples [(1,10),(15,17),...]
        # we need a structured numpy array in this case
        # this gets sketchy here....
        # TODO: raise an error if the tuples are of differen range 
        raise NotImplementedError()
        # return array(
        #     [array([real(res[indices[0]:indices[1],...]), imag(res[indices[0]:indices[1],...])],dtype=float32) for indices in fidx ],
        #     dtype=float32)
    else:
        return array([real(res), imag(res)],dtype=float32)


def get_source_p2(source_mixer, power_spectra, fidx=None, cache_dir=None, num_threads=1):
    """Returns the [Pa^2] values at the reference microphone emitted by the sources 
    contained in the `sources` list of the acoular.SourceMixer object.

    Parameters
    ----------
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the Pa^2 value
    fidx : int, optional
        frequency index at which the Pa^2 value is returned, by default None, meaning that the
        Pa^2 values of all frequency coefficients will be returned
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        The sources Pa^2 values at the reference microphone with shape (nsources,) or (nsources,B/2+1). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    return _get_ref_mic_pow(source_mixer, power_spectra, fidx )

def _get_ref_mic_pow(source_mixer,power_spectra,fidx=None):
    p2 = []
    for src in source_mixer.sources:
        power_spectra.time_data.source=src
        p2.append(real(power_spectra.csm[:][:,0,0]))
    p2 = array(p2)
    if fidx:
        return array([p2[:,indices[0]:indices[1]].sum(1) for indices in fidx])
    else:
        return p2.T

def _get_sourcemap_evaluator(beamformer,sourcemixer,powerspectra,f,num,r):
    if not f:
        if num > 0:
            raise NotImplementedError("calculating over all frequency bands is currently not supported!")
        else:
            sourcemap = array([beamformer.synthetic(f,num=num) for f in beamformer.freq_data.fftfreq()])
        target_p2 = _get_ref_mic_pow(sourcemixer, powerspectra, None).T
        target_loc = array([array(s.loc) for s in sourcemixer.sources],dtype=float32)
    else:
        sourcemap = array([beamformer.synthetic(freq,num=num) for freq in f])
        fidx = [get_frequency_index_range(powerspectra.fftfreq(),freq,num) for freq in f]
        target_p2 = _get_ref_mic_pow(sourcemixer, powerspectra, fidx).T
        target_loc = array([array(s.loc) for s in sourcemixer.sources],dtype=float32)    
    return PlanarSourceMapEvaluator(sourcemap=sourcemap, grid=beamformer.steer.grid, target_loc=target_loc, target_pow=target_p2, r=r)


def get_overall_level_error(beamformer, powerspectra, sourcemixer, r=0.05, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculates the sourcemap with a specified beamformer instance
    of class (or derived class) of type acoular.BeamformerBase and evaluates the overall level error.

    Parameters
    ----------
    beamformer : instance of class acoular.BeamformerBase
        beamformer to calculate the source map feature
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the Pa^2 value
    f : float
        frequency to evaluate
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        sourcemap feature of either shape (nxsteps,nysteps) or (B/2+1,nxsteps,nysteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    se = _get_sourcemap_evaluator(beamformer,sourcemixer,powerspectra,f,num,r)
    return se.get_overall_level_error()


def get_specific_level_error(beamformer, powerspectra, sourcemixer, r=0.05, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculates the sourcemap with a specified beamformer instance
    of class (or derived class) of type acoular.BeamformerBase and evaluates the specific level error.

    Parameters
    ----------
    beamformer : instance of class acoular.BeamformerBase
        beamformer to calculate the source map feature
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the Pa^2 value
    f : float
        frequency to evaluate
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        sourcemap feature of either shape (nxsteps,nysteps) or (B/2+1,nxsteps,nysteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    se = _get_sourcemap_evaluator(beamformer,sourcemixer,powerspectra,f,num,r)
    return se.get_specific_level_error()


def get_inverse_level_error(beamformer, powerspectra, sourcemixer, r=0.05, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculates the sourcemap with a specified beamformer instance
    of class (or derived class) of type acoular.BeamformerBase and evaluates the inverse level error.

    Parameters
    ----------
    beamformer : instance of class acoular.BeamformerBase
        beamformer to calculate the source map feature
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    power_spectra : instance of acoular.PowerSpectra
        object to calculate the Pa^2 value
    f : float
        frequency to evaluate
    cache_dir : str, optional
        directory to store the cache files (only necessary if PowerSpectra.cached=True), 
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution

    Returns
    -------
    numpy.array
        sourcemap feature of either shape (nxsteps,nysteps) or (B/2+1,nxsteps,nysteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    se = _get_sourcemap_evaluator(beamformer,sourcemixer,powerspectra,f,num,r)
    return se.get_inverse_level_error()


# Abstract Base class
class BaseFeature:

    def __init__(self,feature_name):
        self.feature_name = feature_name

    def add_metadata(self,metadata):
        return metadata
                
    def add_feature_funcs(self,feature_funcs):
        return feature_funcs

    def add_encoder_funcs(self,encoder_funcs):
        return encoder_funcs

    def add_feature_names(self,feature_names):
        return feature_names


class SourceMapFeature(BaseFeature):

    def __init__(self,feature_name,beamformer,f,num,cache_dir):
        self.feature_name = feature_name
        self.beamformer = beamformer
        self.f = f
        self.num = num
        self.cache_dir = cache_dir

    def add_metadata(self,metadata):
        metadata['r_diag'] = self.beamformer.r_diag
        metadata['steer_type'] = self.beamformer.steer.steer_type
        metadata['ref'] = self.beamformer.steer.ref    
        return metadata

    def add_feature_funcs(self,feature_funcs):
        feature_funcs[self.feature_name] = (
            get_sourcemap, self.beamformer, self.f, self.num, self.cache_dir)
        return feature_funcs

    def add_encoder_funcs(self, encoder_funcs):
        encoder_funcs[self.feature_name] = float_list_feature
        return encoder_funcs

    def add_feature_names(self,feature_names):
        return feature_names + [self.feature_name]


class RefSourceMapFeature(SourceMapFeature):

    def __init__(self,feature_name,beamformer,sourcemixer,powerspectra,r,f,num,cache_dir):
        self.feature_name = feature_name
        self.beamformer = beamformer
        self.sourcemixer = sourcemixer
        self.powerspectra = powerspectra
        self.r = r
        self.f = f
        self.num = num
        self.cache_dir = cache_dir

    def add_metadata(self,metadata):
        metadata['r_diag'] = self.beamformer.r_diag
        metadata['steer_type'] = self.beamformer.steer.steer_type
        metadata['ref'] = self.beamformer.steer.ref   
        metadata['reference_integration_radius'] = self.r 
        return metadata

    def add_feature_funcs(self,feature_funcs):
        feature_funcs[self.feature_name] = (
            get_sourcemap, self.beamformer, self.f, self.num, self.cache_dir)
        feature_funcs['overall_level_error'] = (
            get_overall_level_error, self.beamformer, self.powerspectra, self.sourcemixer, self.r, self.f, self.num, self.cache_dir)
        feature_funcs['specific_level_error'] = (
            get_specific_level_error, self.beamformer, self.powerspectra, self.sourcemixer, self.r, self.f, self.num, self.cache_dir)
        feature_funcs['inverse_level_error'] = (
            get_inverse_level_error, self.beamformer, self.powerspectra, self.sourcemixer, self.r, self.f, self.num, self.cache_dir)   
        return feature_funcs

    def add_encoder_funcs(self, encoder_funcs):
        encoder_funcs[self.feature_name] = float_list_feature
        encoder_funcs['overall_level_error'] = float_list_feature
        encoder_funcs['specific_level_error'] = float_list_feature
        encoder_funcs['inverse_level_error'] = float_list_feature
        return encoder_funcs

    def add_feature_names(self,feature_names):
        new_names =  [
            self.feature_name,'overall_level_error','specific_level_error','inverse_level_error'
            ]
        return feature_names + new_names

# class RefSBL(SourceMapFeature):
#     def __init__(self,feature_name,spectra_inout,steer,fidx):
#         self.feature_name = feature_name
#         self.spectra_inout = spectra_inout
#         self.steer = steer
#         self.fidx = fidx      

#     def add_metadata(self,metadata):
#         return metadata

#     def add_feature_funcs(self,feature_funcs):
#         feature_funcs[self.feature_name] = (
#             get_SBLmap, self.spectra_inout, self.steer, self.fidx)
#         return feature_funcs

#     def add_encoder_funcs(self, encoder_funcs):
#         encoder_funcs[self.feature_name] = float_list_feature
#         return encoder_funcs

#     def add_feature_names(self,feature_names):
#         return feature_names + [self.feature_name]


class CSMFeature(BaseFeature):

    def __init__(self,feature_name,power_spectra,fidx,cache_dir):
        self.feature_name = feature_name
        self.power_spectra = power_spectra
        self.fidx = fidx
        self.cache_dir = cache_dir

    def add_feature_funcs(self,feature_funcs):
        feature_funcs[self.feature_name] = (
            get_csm, self.power_spectra, self.fidx, self.cache_dir)
        return feature_funcs

    def add_encoder_funcs(self, encoder_funcs):
        encoder_funcs[self.feature_name] = float_list_feature
        return encoder_funcs

    def add_feature_names(self,feature_names):
        return feature_names + [self.feature_name]


# class SpectrogramFeature(BaseFeature):

#     def __init__(self,feature_name,spectra_inout,fidx,cache_dir):
#         self.feature_name = feature_name
#         self.spectra_inout = spectra_inout
#         self.fidx = fidx
#         self.cache_dir = cache_dir

#     def add_feature_funcs(self,feature_funcs):
#         feature_funcs[self.feature_name] = (
#             get_spectrogram, self.spectra_inout, self.fidx, self.cache_dir)
#         return feature_funcs

#     def add_encoder_funcs(self, encoder_funcs):
#         encoder_funcs[self.feature_name] = float_list_feature
#         return encoder_funcs

#     def add_feature_names(self,feature_names):
#         return feature_names + [self.feature_name]


class NonRedundantCSMFeature(CSMFeature):
    
    def add_feature_funcs(self,feature_funcs):
        feature_funcs[self.feature_name] = (
            get_csmtriu, self.power_spectra, self.fidx, self.cache_dir)
        return feature_funcs
