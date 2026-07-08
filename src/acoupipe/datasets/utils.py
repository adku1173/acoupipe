from functools import wraps
from time import time

import acoular as ac
from acoupipe.config import TF_FLAG
from acoupipe.writer import WriteH5Dataset

import numpy as np
import scipy.signal

if TF_FLAG:
    from acoupipe.writer import WriteTFRecord
import logging
from datetime import datetime
from os.path import join
from warnings import warn


def calc_transfer(ir, fs, blocksize, fftfreq, time_axis=-1):
    """
    Compute one-sided transfer functions H(f) from (measured) impulse responses on a
    target rFFT bin grid defined by `blocksize`, returning only `fftfreq`.

    The function enforces that the FFT length is a power-of-two by zero-padding the
    impulse responses to nfft = 2**ceil(log2(max(L, blocksize))), where L is the IR length.

    Parameters
    ----------
    ir : ndarray
        Impulse responses. The time axis is given by `time_axis`.
        Common shapes are (n_channels, n_samples) with time_axis=1,
        or (n_samples, n_channels) with time_axis=0.
    fs : float
        Sampling frequency in Hz.
    blocksize : int
        Target block size defining the desired rFFT bin grid (power-of-two).
        The associated bin centers are f_k = k*fs/blocksize, k=0..blocksize//2.
    fftfreq : array_like of int or float, optional
        Desired frequency bins to return.

        - If integer dtype: interpreted as rFFT bin indices on the `blocksize` grid,
          i.e., k in [0, blocksize//2]. This case is exact because both blocksize and
          nfft are powers-of-two, hence nfft/blocksize is integer and mapping is exact.
        - If float dtype: interpreted as frequencies in Hz. Values are mapped to the
          nearest rFFT bin of the computed FFT grid; a ValueError is raised if the
          frequency is not (approximately) on the grid.

        If None, all one-sided bins of the computed FFT are returned.
    time_axis : int, optional
        Axis of `ir` corresponding to time samples. Default: -1.

    Returns
    -------
    H_sel : ndarray (complex)
        One-sided transfer function values at the requested bins. The returned array
        has the same shape as `ir`, except the time axis is replaced by a frequency axis.
        If fftfreq is None, this frequency axis has length nfft//2+1.
        Otherwise, it has length len(fftfreq).
    """
    L = ir.shape[time_axis]

    # Select FFT length: power-of-two >= max(L, blocksize)
    nfft = 1 << (max(L, blocksize) - 1).bit_length()

    # Zero-pad IRs to nfft (never truncate)
    if nfft > L:
        pad_width = [(0, 0)] * ir.ndim
        pad_width[-1] = (0, nfft - L)
        ir = np.pad(ir, pad_width, mode='constant')

    # One-sided FFT with window matching the (zero-padded) length (i.e., use full nfft)
    H = np.fft.rfft(ir, n=nfft, axis=-1)  # shape: (..., nfft//2+1)
    np.fft.rfftfreq(nfft, d=1.0 / fs)

    if np.any(fftfreq < 0.0) or np.any(fftfreq > fs / 2.0):
        msg = 'Requested frequencies must satisfy 0 <= f <= fs/2.'
        raise ValueError(msg)

    # Map to nearest bin; require exact (within tolerance) grid match
    idx_float = fftfreq * nfft / fs
    idx = np.rint(idx_float).astype(int)
    tol = 1e-9
    if np.max(np.abs(idx_float - idx)) > tol:
        msg = (
            'Some requested frequencies are not on the FFT grid. '
            'Provide integer bin indices (recommended), or choose f that matches k*fs/nfft.'
        )
        raise ValueError(msg)

    H_sel = np.take(H, idx, axis=-1)
    return H_sel.T


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
    logging.basicConfig(level=logging.INFO)  # root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    pipeline_log = logging.FileHandler(fname, mode='w')  # log everything to file
    pipeline_log.setFormatter(
        logging.Formatter('%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d-%(message)s', datefmt='%Y-%m-%d,%H:%M:%S'),
    )
    logger.addHandler(pipeline_log)  # attach handler to the root logger


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
    alpha = np.sin(omega_low) * np.sqrt((A**2 + 1) * ((1 / q) - 1) + 2 * A)

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
    alpha = np.sin(omega_high) * np.sqrt((A**2 + 1) * ((1 / q) - 1) + 2 * A)

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


def get_frequency_index_range(freq, f, num):
    """Return the left and right indices that define the frequency range to integrate over.

    Parameters
    ----------
    freq : numpy.array
        frequency vector (can be determined by evaluating `freqdata()` method
        at a `acoular.PowerSpectra` instance)
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
        ind = np.searchsorted(freq, f)
        if ind >= len(freq):
            warn(
                f'Queried frequency ({f:g} Hz) not in resolved frequency range. Returning zeros.',
                Warning,
                stacklevel=2,
            )
            ind = None
        else:
            if freq[ind] != f:
                warn(
                    f'Queried frequency ({f:g} Hz) not in set of '
                    'discrete FFT sample frequencies. '
                    f'Using frequency {freq[ind]:g} Hz instead.',
                    Warning,
                    stacklevel=2,
                )
        return (ind, ind + 1)
    # fractional octave band
    if isinstance(num, list):
        f1 = num[0]
        f2 = num[-1]
    else:
        f1 = f * 2.0 ** (-0.5 / num)
        f2 = f * 2.0 ** (+0.5 / num)
    ind1 = np.searchsorted(freq, f1)
    ind2 = np.searchsorted(freq, f2)
    if ind1 == ind2:
        warn(
            f'Queried frequency band ({f1:g} to {f2:g} Hz) does not '
            'include any discrete FFT sample frequencies. '
            'Returning zeros.',
            Warning,
            stacklevel=2,
        )
    return (ind1, ind2)


def set_pipeline_seeds(pipeline, start_idx, size, dataset='training'):
    """Create the random seed list for each sampler object held by the pipeline.

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
    if dataset == 'training':
        off = 0
    elif dataset == 'validation':
        # we assume that the training set will never be larger than 1e12 samples
        off = int(1e12)  # a general offset to ensure that validation and training seeds never match
    elif dataset == 'test':
        off = int(1e21)
    soff = int(1e7)  # offset to ensure that seeds of sampler object doesn't match
    pipeline.random_seeds = {
        i: range(off + (i * soff) + start_idx, off + (i * soff) + size + start_idx)
        for i in list(pipeline.sampler.keys())
    }


def set_filename(writer, path='.', *args):
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
    name = f'{args[0]}'
    for arg in args[1:]:
        name += f'_{arg}'
    name += f'_{datetime.now().strftime("%d-%b-%Y")}'  # noqa: DTZ005
    if isinstance(writer, WriteH5Dataset):
        name += '.h5'
    if TF_FLAG and isinstance(writer, WriteTFRecord):
        name += '.tfrecord'
    writer.name = join(path, name)


def complex_to_real(func):
    def complex_to_real_wrapper(*args, **kwargs):
        a = func(*args, **kwargs)
        return np.concatenate([np.real(a)[..., np.newaxis], np.imag(a)[..., np.newaxis]], axis=-1)

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
    if isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_point_sources_recursively(source.source)
        for s in source.sources:
            sources += get_point_sources_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return []
    return sources


def _get_signals_recursively(source, signals):
    if hasattr(source, 'signal') and isinstance(source.signal, ac.SignalGenerator):
        signals.append(source.signal)
    elif hasattr(source, 'sources') or hasattr(source, 'sources') and isinstance(source, ac.SamplesGenerator):
        if hasattr(source, 'sources'):
            for s in source.sources:
                signals = _get_signals_recursively(s, signals)
        if hasattr(source, 'source'):
            signals = _get_signals_recursively(source.source, signals)
    return signals


def get_all_source_signals(source_list):
    """Get all signals from a list of `acoular.SamplesGenerator` derived objects.

    Parameters
    ----------
    source_list : list
        list of `acoular.SamplesGenerator` derived objects


    Returns
    -------
    list
        list of all `acoular.SignalGenerator` derived objects
    """
    signals = []
    for source in source_list:
        if not isinstance(source, ac.SamplesGenerator):
            msg = 'source must be of type `acoular.SamplesGenerator`'
            raise ValueError(msg)
        signals = _get_signals_recursively(source, signals)
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
    if isinstance(source, ac.SourceMixer):
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.Mixer):
        sources += get_uncorrelated_noise_source_recursively(source.source)
        for s in source.sources:
            sources += get_uncorrelated_noise_source_recursively(s)
    elif isinstance(source, ac.UncorrelatedNoiseSource):
        return [source]
    return sources


# Without the use of this decorator factory (wraps), the name of the
# function 'f' would have been 'wrap', and the docstring of the original f() would have been lost.
def log_execution_time(f):
    """Log execution time during feature calculation."""

    @wraps(f)
    def wrap(self, *args, **kw):
        self.logger.info(f'id {self._idx}: start task.')
        start = time()
        result = f(self, *args, **kw)
        end = time()
        self.logger.info(f'id {self._idx}: finished task.')
        # self.logger.info(f"{f.__name__} args:[{args}] took: {end-start:.32f} sec")
        self.logger.info(f'id {self._idx}: executing task took: {end - start:.32f} sec')
        return result

    return wrap
