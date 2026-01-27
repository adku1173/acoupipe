import pyroomacoustics as pra

def get_ir_pyroom_acoustics(sample_freq, room_dim, mloc, sloc, rt60, use_rand_ism=False):
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
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
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
