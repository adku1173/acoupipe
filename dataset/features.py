from acoular import config
from numpy import zeros, array, float32, concatenate, real, imag, triu_indices
import numba

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
        sourcemap feature of either shape (nxsteps,nysteps) or (B/2+1,nxsteps,nysteps). 
        B: Blocksize of the FFT.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if not f:
        return array([beamformer.synthetic(f,num=num) for f in beamformer.freq_data.fftfreq()])
    else:
        return beamformer.synthetic(f,num=num) 
    


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
        The cross-spectral matrix with shape (2,M,M) or (2,B/2+1,M,M) if fidx=None. 
        B: Blocksize of the FFT. M: Number of microphones.Real values will 
        be stored at the first entry of the first dimension. 
        Imaginary values will be stored at the second entry of the first dimension.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    if fidx:
        return array([real(power_spectra.csm[fidx,:,:]), imag(power_spectra.csm[fidx,:,:])],dtype=float32)
    else:
        return array([real(power_spectra.csm), imag(power_spectra.csm)],dtype=float32)



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
        The cross-spectral matrix with shape (M,M) or (B/2+1,M,M) if fidx=None. 
        B: Blocksize of the FFT. M: Number of microphones.Real values will 
        be stored at the upper triangular matrix. 
        Imaginary values will be stored at the lower triangular matrix.
    """
    numba.set_num_threads(num_threads)
    if cache_dir:
        config.cache_dir = cache_dir
    num_mics = power_spectra.numchannels
    if not fidx:
        csm = []
        for i in range(power_spectra.csm.shape[0]):
            csm_recover_real = zeros((64,64),dtype=float32)
            csm_recover_imag = zeros((64,64),dtype=float32)
            csm_real = real(power_spectra.csm[i])[triu_indices(num_mics)]
            csm_imag = imag(power_spectra.csm[i])[triu_indices(num_mics)]
            csm_recover_real[triu_indices(num_mics)] = csm_real # add real part at upper triangular matrix
            csm_recover_imag[triu_indices(num_mics)] = csm_imag
            csm.append(csm_recover_real + csm_recover_imag.T)
        return array([csm],dtype=float32).squeeze()
    else:
        csm_recover_real = zeros((num_mics,num_mics),dtype=float32)
        csm_recover_imag = zeros((num_mics,num_mics),dtype=float32)
        csm_real = real(power_spectra.csm[fidx])[triu_indices(num_mics)]
        csm_imag = imag(power_spectra.csm[fidx])[triu_indices(num_mics)]
        csm_recover_real[triu_indices(num_mics)] = csm_real # add real part at upper triangular matrix
        csm_recover_imag[triu_indices(num_mics)] = csm_imag
        return (csm_recover_real + csm_recover_imag.T)



def get_source_p2(source_mixer, power_spectra, fidx=None, nsources=16, cache_dir=None, num_threads=1):
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
    nsources : int, optional
        number of sources, by default 16. If less than 16 sources are present, the remaining 
        entries will be padded with zeros.
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
    p2 = get_ref_mic_pow(source_mixer, power_spectra, fidx)
    if cache_dir:
        config.cache_dir = cache_dir
    if fidx:
        p2_zeros = zeros((nsources-len(source_mixer.sources)))
    else:
        nfreqs=p2.shape[1]
        p2_zeros = zeros((nsources-len(source_mixer.sources),nfreqs))
    return concatenate((p2,p2_zeros),axis=0)



def get_source_loc(source_mixer,nsources=16,dim=2):
    """Returns the location of the sources contained by the `sources` attribute
    of the acoular.SourceMixer object.
     
    Parameters
    ----------
    source_mixer : instance of acoular.SourceMixer
        SourceMixer object holding PointSource objects
    nsources : int, optional
        number of sources, by default 16. If less than 16 sources are present, the remaining 
        loc entries will be padded with zeros.
    dim : int, optional
        dimension of the returned source positions, by default 2 (x,y) values.

    Returns
    -------
    numpy.array
        The source locations of shape (nsources,dim)
    """
    loc_zeros = zeros((nsources-len(source_mixer.sources),dim))
    loc = array([array(s.loc) for s in source_mixer.sources],dtype=float32)[:,:dim] # ignore third dimension
    return concatenate((loc,loc_zeros),axis=0)

def get_ref_mic_pow(source_mixer,power_spectra,fidx=None):
    p2 = []
    for src in source_mixer.sources:
        power_spectra.time_data.source=src
        p2.append(real(power_spectra.csm[:][:,0,0]))
    if fidx:
        return array(p2)[:,fidx]
    else:
        return array(p2)