from acoular import config

from acoupipe.config import TF_FLAG

if TF_FLAG:
    pass
import warnings

import numba
from numpy import array, float32, imag, newaxis, real, triu_indices, zeros
from numpy.linalg import eigh
from threadpoolctl import threadpool_limits

from acoupipe.datasets.helper import complex_to_real

#from .sbl import SBL, Options
warnings.filterwarnings("ignore") # suppress pickling warnings

def get_sourcemap(beamformer, f=None, num=0, cache_dir=None, num_threads=1):
    """Calculate the sourcemap with a specified beamformer instance of class (or derived class) of type acoular.BeamformerBase.

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
    return sm


@complex_to_real
def get_csm(freq_data, fidx=None, cache_dir=None, num_threads=1):
    """Calculate the cross-spectral matrix (CSM) from time data.

    Parameters
    ----------
    freq_data : instance of class acoular.PowerSpectra
        power spectra to calculate the csm feature
    fidx : list of tuples, optional     
        list of tuples containing the start and end indices of the frequency bands to be considered,
        by default None
    cache_dir : str, optional   
        directory to store the cache files (only necessary if PowerSpectra.cached=True),
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution
    

    Returns
    -------
    numpy.array
        The cross-spectral matrix with shape (1,M,M,2) or (B/2+1,M,M,2) if fidx=None.
        B: Blocksize of the FFT. M: Number of microphones. Real values will
        be stored at the first entry of the last dimension.
        Imaginary values are stored at the second entry of the last dimension.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    csm = freq_data.csm[:]
    if fidx:
        csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)
    return csm

@complex_to_real
def get_eigmode(freq_data,fidx=None,cache_dir=None,num_threads=1):
    """Calculate the eigenvalue-scaled eigenvectors of the cross-spectral matrix (CSM) from time data.

    Parameters
    ----------
    freq_data : instance of class acoular.PowerSpectra
        power spectra to calculate the csm feature
    fidx : list of tuples, optional     
        list of tuples containing the start and end indices of the frequency bands to be considered,
        by default None
    cache_dir : str, optional   
        directory to store the cache files (only necessary if PowerSpectra.cached=True),
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution
    

    Returns
    -------
    numpy.array
        The eigenvalue scaled eigenvectors with shape (1,M,M,2) or (B/2+1,M,M,2) if fidx=None.
        B: Blocksize of the FFT. M: Number of microphones. Real values will
        be stored at the first entry of the last dimension.
        Imaginary values are stored at the second entry of the last dimension.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    csm = freq_data.csm[:]
    if fidx:
        csm = array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=complex)           
    with threadpool_limits(limits=1, user_api="blas"): # limit the number of threads used by numpy LAPACK routines
        # using a single thread improved throughput even in single task mode
        eva, eve = eigh(csm)
    return eva[:,newaxis,:]*eve[:]


def get_nonredundant_csm(freq_data, fidx=None, cache_dir=None, num_threads=1):
    """Calculate the non-redundant cross-spectral matrix (CSM) from time data.

    According to:
    Paolo Castellini, Nicola Giulietti, Nicola Falcionelli, Aldo Franco Dragoni, Paolo Chiariotti,
        A neural network based microphone array approach to grid-less noise source localization,
        Applied Acoustics, Volume 177, 2021.

    Parameters
    ----------
    freq_data : instance of class acoular.PowerSpectra
        power spectra to calculate the csm feature
    fidx : list of tuples, optional     
        list of tuples containing the start and end indices of the frequency bands to be considered,
        by default None
    cache_dir : str, optional   
        directory to store the cache files (only necessary if PowerSpectra.cached=True),
        by default None
    num_threads : int, optional
        the number of threads used by numba during parallel execution
    

    Returns
    -------
    numpy.array
        The non-redundant cross-spectral matrix with shape (1,M,M,1) or (B/2+1,M,M,1) if fidx=None.
        B: Blocksize of the FFT. M: Number of microphones. 
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if not fidx:
        return _transform_csm(freq_data.csm)
    else:  
        csm = _transform_csm(freq_data.csm)  
        return array([csm[indices[0]:indices[1]].sum(0) for indices in fidx],dtype=float32)


def get_source_p2(source_mixer, freq_data, fidx=None, cache_dir=None, num_threads=1):
    """Return the [Pa^2] values at the reference microphone emitted by the sources contained by a acoular.SourceMixer object.

    Parameters
    ----------
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    freq_data : instance of acoular.PowerSpectra
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
    return _get_ref_mic_pow(source_mixer, freq_data, fidx )


def _get_ref_mic_pow(source_mixer,freq_data,fidx=None):
    p2 = []
    for src in source_mixer.sources:
        freq_data.time_data.source=src
        p2.append(real(freq_data.csm[:][:,0,0]))
    p2 = array(p2)
    if fidx:
        return array([p2[:,indices[0]:indices[1]].sum(1) for indices in fidx])
    else:
        return p2.T

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


# def get_eigmap(beamformer, n=16, f=None, num=0, cache_dir=None, num_threads=1):
#     """Calculates the sourcemap with a specified beamformer instance
#     of class (or derived class) of type acoular.BeamformerBase.

#     Parameters
#     ----------
#     beamformer : instance of class acoular.BeamformerBase
#         beamformer to calculate the source map feature
#     n : int
#         eigenvalue components to calculate (n strongest)
#     f : float
#         frequency to evaluate
#     num : integer
#         Controls the width of the frequency bands considered; defaults to
#         0 (single frequency line).
#         ===  =====================
#         num  frequency band width
#         ===  =====================
#         0    single frequency line
#         1    octave band
#         3    third-octave band
#         n    1/n-octave band
#         ===  =====================
#     cache_dir : str, optional
#         directory to store the cache files (only necessary if PowerSpectra.cached=True), 
#         by default None
#     num_threads : int, optional
#         the number of threads used by numba during parallel execution

#     Returns
#     -------
#     numpy.array
#         sourcemap feature of either shape (n,nxsteps,nysteps) or (n,B/2+1,n,nxsteps,nysteps). 
#         B: Blocksize of the FFT.
#     """
#     numba.set_num_threads(num_threads)
#     if cache_dir:
#         config.cache_dir = cache_dir
#     if not f:
#         eig_maps = []
#         for i in range(n):
#             beamformer.n = (-1)-i # get the sourcemap for the n largest eigenvalues
#             eig_maps.append([beamformer.synthetic(f,num=num) for f in beamformer.freq_data.fftfreq()])
#         return array(eig_maps)
#     else:
#         eig_maps = []
#         for i in range(n):
#             beamformer.n = (-1)-i # get the sourcemap for the n largest eigenvalues
#             eig_maps.append(beamformer.synthetic(f,num=num))
#         return array(eig_maps) # sort from largest to smallest 


# def get_SBLmap(spectra_inout, steer, fidx=None):
#     """
#     """
#     options = Options(convergence_error=10 ** (-8), gamma_range=10 ** (-4), convergence_maxiter=5000, 
#               convergence_min_iteration=1, status_report=1, fixedpoint=1, Nsource=len(spectra_inout.source.sources), flag=0)    
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

# def get_spectrogram(spectra_inout, fidx=None, cache_dir=None, num_threads=1):
#     """Calculates the cross-spectral matrix (CSM). 

#     Parameters
#     ----------
#     spectra_inout : instance of spectacoular.SpectraInOut
#         object to calculate the CSM
#     fidx : int, optional
#         frequency index at which the CSM is returned, by default None, meaning that the
#         CSM for all frequency coefficients will be returned
#     cache_dir : str, optional
#         directory to store the cache files (only necessary if PowerSpectra.cached=True), 
#         by default None
#     num_threads : int, optional
#         the number of threads used by numba during parallel execution

#     Returns
#     -------
#     numpy.array
#         The complex spectrogram matrix with shape (2,B/2+1,T,M) if fidx=None. 
#         B: Blocksize of the FFT. M: Number of microphones.Real values will 
#         be stored at the first entry of the first dimension. 
#         Imaginary values will be stored at the second entry of the first dimension.
#     """
#     numba.set_num_threads(num_threads)
#     if cache_dir:
#         config.cache_dir = cache_dir
#     res = array(list(spectra_inout.result())).swapaxes(0,1)        
#     if fidx: # list with freq index tuples [(1,10),(15,17),...]
#         # we need a structured numpy array in this case
#         # this gets sketchy here....
#         # TODO: raise an error if the tuples are of differen range 
#         raise NotImplementedError()
#         # return array(
#         #     [array(
#               [real(res[indices[0]:indices[1],...]), imag(res[indices[0]:indices[1],...])],dtype=float32) for indices in fidx ],
#         #     dtype=float32)
#     else:
#         return array([real(res), imag(res)],dtype=float32)

