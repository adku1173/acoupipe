from functools import wraps
from time import time

import acoular as ac
import numpy as np
import scipy.signal

from acoupipe.config import TF_FLAG
from acoupipe.writer import WriteH5Dataset

if TF_FLAG:
    from acoupipe.writer import WriteTFRecord
import logging
from datetime import datetime
from os.path import join
from warnings import warn

from numpy import concatenate, imag, newaxis, real, searchsorted


def tqdm_hook(t):
    """Wrap tqdm instance according to https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """Update tqdm instance with bytes transferred and total size.

        Parameters
        ----------
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def _handle_log(fname):
    logging.basicConfig(level=logging.INFO) # root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    pipeline_log = logging.FileHandler(fname,mode="w") # log everything to file
    pipeline_log.setFormatter(logging.Formatter(
        "%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d-%(message)s", datefmt="%Y-%m-%d,%H:%M:%S"))
    logger.addHandler(pipeline_log) # attach handler to the root logger


def generate_uniform_parametric_eq(num_points, max_order, rng):
    """Generate a random parametric EQ cascase.

    Method according to [Nercessian 2020](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_7.pdf).
    This method is part of the `IIRNet project <https://github.com/csteinmetz1/IIRNet>`_.
    License: Apache License 2.0

    Returns
    -------
        coef
        mag
        phs
    """
    zeros = []
    poles = []
    sos_holder = []
    num_peaks = (max_order) // 2 - 2  # Number of peaking filters to use paper=10

    omega_low = rng.uniform(low=0.0, high=np.pi)
    g = rng.uniform(low=-10.0, high=10.0)
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_low) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) - (A - 1) * np.cos(omega_low) + alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_low))
    b2 = A * ((A + 1) - (A - 1) * np.cos(omega_low) - alpha)

    a0 = (A + 1) + (A - 1) * np.cos(omega_low) + alpha
    a1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_low))
    a2 = (A + 1) + (A - 1) * np.cos(omega_low) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    omega_high = rng.uniform(low=0.0, high=np.pi)
    g = rng.uniform(low=-10.0, high=10.0)
    q = rng.uniform(low=0.1, high=1.0)
    A = np.power(10, g / 40)
    alpha = np.sin(omega_high) * np.sqrt((A ** 2 + 1) * ((1 / q) - 1) + 2 * A)

    b0 = A * ((A + 1) + (A - 1) * np.cos(omega_high) + alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(omega_high))
    b2 = A * ((A + 1) + (A - 1) * np.cos(omega_high) - alpha)

    a0 = (A + 1) - (A - 1) * np.cos(omega_high) + alpha
    a1 = 2 * A * ((A - 1) - (A + 1) * np.cos(omega_high))
    a2 = (A + 1) - (A - 1) * np.cos(omega_high) - alpha

    sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
    sos_holder.append(sos_poly)
    num_poly = np.asarray([b0, b1, b2])
    zeros.append(num_poly)
    den_poly = np.asarray([a0, a1, a2])
    poles.append(den_poly)

    ##Peaking Filters
    for _jj in range(num_peaks):
        omega = rng.uniform(low=0.0, high=np.pi)
        g = rng.uniform(low=-10, high=10)
        q = rng.uniform(low=0.1, high=3.0)

        alpha = np.sin(omega) / (2 * q)
        A = np.power(10, g / 40)

        b0 = 1 + (alpha * A)
        b1 = -2 * np.cos(omega)
        b2 = 1 - (alpha * A)

        a0 = 1 + (alpha / A)
        a1 = -2 * np.cos(omega)
        a2 = 1 - (alpha / A)

        sos_poly = np.asarray([b0, b1, b2, a0, a1, a2])
        sos_holder.append(sos_poly)
        num_poly = np.asarray([b0, b1, b2])
        zeros.append(num_poly)
        den_poly = np.asarray([a0, a1, a2])
        poles.append(den_poly)

    sos = np.vstack(sos_holder)
    my_norms = sos[:, 3]
    sos = sos / my_norms[:, None]  ##sosfreqz requires sos[:,3]=1
    w, h = scipy.signal.sosfreqz(sos, worN=num_points)
    return h, sos


def get_frequency_index_range(freq,f,num):
    """Return the left and right indices that define the frequency range to integrate over.

    Parameters
    ----------
    freq : numpy.array
        frequency vector (can be determined by evaluating `freqdata()` method at a `acoular.PowerSpectra` instance)
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)

    Returns
    -------
    tuple
        left and right index that belongs to the frequency of interest
    """
    if num == 0:
        # single frequency line
        ind = searchsorted(freq, f)
        if ind >= len(freq):
            warn("Queried frequency (%g Hz) not in resolved "
                            "frequency range. Returning zeros." % f,
                            Warning, stacklevel = 2)
            ind = None
        else:
            if freq[ind] != f:
                warn("Queried frequency (%g Hz) not in set of "
                        "discrete FFT sample frequencies. "
                        "Using frequency %g Hz instead." % (f,freq[ind]),
                        Warning, stacklevel = 2)
        return (ind,ind+1)
    else:
        # fractional octave band
        if isinstance(num,list):
            f1=num[0]
            f2=num[-1]
        else:
            f1 = f*2.**(-0.5/num)
            f2 = f*2.**(+0.5/num)
        ind1 = searchsorted(freq, f1)
        ind2 = searchsorted(freq, f2)
        if ind1 == ind2:
            warn("Queried frequency band (%g to %g Hz) does not "
                    "include any discrete FFT sample frequencies. "
                    "Returning zeros." % (f1,f2),
                    Warning, stacklevel = 2)
        return (ind1,ind2)


def set_pipeline_seeds(pipeline,start_idx,size,dataset="training"):
    """Create the random seed list for each of the sampler objects that is held by the pipeline object.

    Parameters
    ----------
    pipeline : instance of class BasePipeline
        the pipeline object holding the sampler classes
    start_idx : int
        start index to be calculated by the pipeline
    size : int
        number of samples to be yielded by the pipeline
    dataset : str, optional
        the data set type, by default "training". Choose from ["training","validation"]
    """
    if dataset=="training":
        off = 0
    elif dataset=="validation":
        off = int(1e16) # a general offset to ensure that validation and training seeds never match (max seed is 2**32)
    elif dataset == "test":
        off = int(1e24)
    soff = int(1e7) # offset to ensure that seeds of sampler object doesn't match
    pipeline.random_seeds = {i : range(off+(i*soff)+start_idx, off+(i*soff)+size+start_idx) for i in list(pipeline.sampler.keys())}



def set_filename(writer,path=".",*args):
    """Set the filename of the dataset.

    Parameters
    ----------
    writer : instance of class BaseWriteDataset
        the writer object holding the filename
    path : str, optional
        the path to the dataset, current directory by default
    *args : str
        concatenated strings to be used as the filename
    """
    name = f"{args[0]}"
    for arg in args[1:]:
        name += f"_{arg}"
    name += f"_{datetime.now().strftime('%d-%b-%Y')}"
    if isinstance(writer,WriteH5Dataset):
        name += ".h5"
    if TF_FLAG:
        if isinstance(writer,WriteTFRecord):
            name += ".tfrecord"
    writer.name=join(path,name)


def complex_to_real(func):
    def complex_to_real_wrapper(*args,**kwargs):
        a = func(*args,**kwargs)
        return concatenate(
            [real(a)[...,newaxis],
            imag(a)[...,newaxis]],axis=-1)
    return complex_to_real_wrapper


def get_point_sources_recursively(source):
    """Recursively get all point sources from a `acoular.TimeInOut` object.

    Parameters
    ----------
    source : instance of class `acoular.TimeInOut`
        the source object

    Returns
    -------
    list
        list of all sources
    """
    sources = []
    if isinstance(source, ac.PointSource):
        return [source]
    elif isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_point_sources_recursively(source.source)
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return []
    return sources


def get_signals_recursively(source):
    """Recursively get all signals from a `acoular.PointSource` object.

    Parameters
    ----------
    source : instance of class `acoular.TimeInOut`
        the source object

    Returns
    -------
    list
        list of all signals
    """
    signals = []
    if isinstance(source, ac.PointSource):
        signal = source.Signal
        if not isinstance(signal, ac.SignalGenerator):
            raise ValueError("Signal must be of type `acoular.SignalGenerator`")
        return [signal]
    elif isinstance(source, ac.SourceMixer):
        for s in source.sources:
            signals += get_signals_recursively(s)
    elif isinstance(source, ac.Mixer):
        signals += get_signals_recursively(source.source)
        for s in source.sources:
            signals += get_signals_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return []
    return signals

def get_uncorrelated_noise_source_recursively(source):
    """Recursively get all uncorrelated noise sources from a `acoular.TimeInOut` object.

    Parameters
    ----------
    source : instance of class `acoular.TimeInOut`
        the source object

    Returns
    -------
    list
        list of all uncorrelated noise sources
    """
    sources = []
    if isinstance(source, ac.PointSource):
        return []
    elif isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_uncorrelated_noise_source_recursively(source.source)
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return [source]
    return sources

def blockwise_transfer(ir,blocksize):
    """Calculate the transfer function of an impulse response in a blockwise manner.

    Parameters
    ----------
    ir : ndarray, shape (n_channels,n_samples)
        Impulse response.
    blocksize : int, optional
        Block size for the FFT. The default is 256.

    Returns
    -------
    tf : ndarray, shape (n_channels, n_samples)
        Power spectrum of the impulse response.
    """
    n_channels, n_samples = ir.shape
    pad = blocksize - n_samples % blocksize
    ir = np.pad(ir,((0,0),(0,pad)))
    n_blocks = ir.shape[-1] // blocksize
    tf = np.zeros((n_channels, blocksize//2+1), dtype=complex)
    for i in range(n_blocks):
        tf += np.fft.rfft(ir[:,i*blocksize:(i+1)*blocksize], axis=1)
    return tf


# Without the use of this decorator factory (wraps), the name of the
# function 'f' would have been 'wrap', and the docstring of the original f() would have been lost.
def log_execution_time(f):
    """Log execution time during feature calculation."""
    @wraps(f)
    def wrap(self, *args, **kw):
        self.logger.info("id %i: start task." %self._idx)
        start = time()
        result = f(self, *args, **kw)
        end = time()
        self.logger.info("id %i: finished task." %self._idx)
        # self.logger.info("%r args:[%r] took: %2.32f sec" % \
        # (f.__name__,args,end-start))
        self.logger.info("id %i: executing task took: %2.32f sec" % \
        (self._idx,end-start))
        return result
    return wrap
