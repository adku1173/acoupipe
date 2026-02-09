import gpuRIR as grir
import numpy as np
import pyroomacoustics as pra


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


def get_ir_gpurir(sample_freq, room_dim, mloc, sloc, rt60, c=343.0, **kwargs):
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
        rirs.append(rir.squeeze())
    return rirs
