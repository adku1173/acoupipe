import importlib

import numpy as np

UNSUPPORTED_IR_MESSAGE = (
    'Impulse-response-backed synthetic datasets are an unsupported developer-only feature. '
    "Install the optional dependency 'pyroomacoustics' manually to use them."
)


def _load_pyroomacoustics():
    try:
        return importlib.import_module('pyroomacoustics')
    except ImportError as exc:
        raise ImportError(UNSUPPORTED_IR_MESSAGE) from exc


def _load_gpurir():
    try:
        return importlib.import_module('gpuRIR')
    except (ImportError, OSError) as exc:
        msg = 'gpuRIR is not available. Install it manually to use the optional GPU IR backend.'
        raise ImportError(msg) from exc


def require_ir_support():
    """Validate that the unsupported developer-only IR dataset support is installed."""
    _load_pyroomacoustics()


def sabine(room_size=None, alpha=None):
    V = np.prod(room_size)
    A = (
        (alpha[0] + alpha[1]) * room_size[1] * room_size[2]
        + (alpha[2] + alpha[3]) * room_size[0] * room_size[2]
        + (alpha[4] + alpha[5]) * room_size[0] * room_size[1]
    )
    return 0.1611 * V / A


def get_ir_pyroom_acoustics(sample_freq, room_dim, mloc, sloc, rt60, use_rand_ism=False, c=343.0):
    """Get impulse responses using pyroomacoustics.

    Parameters
    ----------
    room_dim : array-like
        Dimensions of the room [length, width, height].
    mloc : array-like
        Microphone locations, shape (3, num_mics).
    sloc : array-like
        Source locations, shape (3, num_sources).
    rt60 : float
        Reverberation time in seconds.
    sample_freq : float
        Sampling frequency in Hz.
    use_rand_ism : bool, optional
        Whether to use random image source method, by default True.

    Returns
    -------
    rir : list of list of np.ndarray
        RIRs for each microphone-source pair.
    """
    pra = _load_pyroomacoustics()
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim, c=c)
    room = pra.ShoeBox(
        room_dim,
        fs=int(sample_freq),
        materials=pra.Material(e_absorption),
        max_order=max_order,
        use_rand_ism=use_rand_ism,
    )
    room.sources = [pra.SoundSource(loc) for loc in sloc.T]
    room.add_microphone_array(pra.MicrophoneArray(mloc, fs=room.fs))
    room.compute_rir()
    return room.rir


def get_ir(sample_freq, room_dim, mloc, sloc, rt60, c=343.0, **kwargs):
    """Get impulse responses for the unsupported developer-only synthetic IR dataset."""
    try:
        return get_ir_gpurir(sample_freq, room_dim, mloc, sloc, rt60, c=c, **kwargs)
    except ImportError:
        return get_ir_pyroom_acoustics(sample_freq, room_dim, mloc, sloc, rt60, c=c)


def get_ir_gpurir(sample_freq, room_dim, mloc, sloc, rt60, c=343.0, **kwargs):
    pra = _load_pyroomacoustics()
    grir = _load_gpurir()
    alpha, _ = pra.inverse_sabine(rt60, room_dim, c=c)
    beta = np.sqrt(1 - alpha**2).tolist()
    beta_list = [beta] * 6
    Tdiff = grir.att2t_SabineEstimator(15, rt60)  # time to consider for diffuse reverberation tail
    rirs = []
    for j in range(mloc.shape[1]):
        rir = grir.simulateRIR(
            room_sz=room_dim,
            beta=beta_list,
            pos_src=sloc.T,
            pos_rcv=mloc[:, j][np.newaxis],
            nb_img=grir.t2n(Tdiff, room_dim),  # Number of image sources in each dimension
            Tmax=rt60,
            fs=sample_freq,
            Tdiff=Tdiff,
            c=c,
            **kwargs,
        )
        # remove microphone dimension (be carfull if only one source is simulated, as the output shape will be different)
        rir = rir.squeeze()
        # at least 2D (num_sources, num_samples) should be preserved
        if rir.ndim == 1:
            rir = rir[np.newaxis, :]
        rirs.append(rir)
    return rirs


def calc_transfer(ir, fs, blocksize, fftfreq, time_axis=-1):
    """Compute one-sided transfer functions H(f) from (measured) impulse responses.

    The transfer functions are evaluated on a target rFFT bin grid defined by `blocksize`,
    returning only `fftfreq`.

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
