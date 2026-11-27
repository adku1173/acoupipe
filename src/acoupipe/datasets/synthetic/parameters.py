"""Synthetic dataset parameters."""

from acoupipe.datasets.base.parameters import ParameterSet
from acoupipe.datasets.synthetic.micgeom import tub_vogel64_ap1


class SyntheticParameters(ParameterSet):
    """
    Shallow simulation parameters for synthetic datasets.

    The synthetic schema stores only mutable simulation state that is not
    derivable from other parameters. Processing-engine settings such as Welch or
    Wishart mode, FFT settings, and Feature-specific parameters are deliberately
    excluded from this class. Source locations are exposed canonically as
    ``source_locations``; legacy feature names such as ``loc`` are handled when
    retrieving Features, not by the ParameterSet itself.

    Parameters
    ----------
    c : float, optional
        Simulation speed of sound in metres per second. Defaults to ``343.0``.
    fs : float, optional
        Sampling frequency in hertz. Defaults to ``13720.0``.
    signal_length : float, optional
        Signal length in seconds. Defaults to ``5.0``.
    mics : object, optional
        Microphone positions, typically an array with shape ``(3, M)``. If not
        supplied, the legacy ``tub_vogel64_ap1`` geometry is copied.
    source_locations : object, optional
        Active source positions. No eager shape validation is performed.
    source_rms : object, optional
        Active source RMS values. No eager shape validation is performed.
    snr_db : object, optional
        Requested microphone signal-to-noise ratio in decibels.
    **parameters
        Additional extension parameters.
    """

    def __init__(
        self,
        c=343.0,
        fs=13720.0,
        signal_length=5.0,
        mics=None,
        source_locations=None,
        source_rms=None,
        snr_db=None,
        **parameters,
    ):
        if mics is None:
            mics = tub_vogel64_ap1.copy()
        super().__init__(
            c=c,
            fs=fs,
            signal_length=signal_length,
            mics=mics,
            source_locations=source_locations,
            source_rms=source_rms,
            snr_db=snr_db,
            **parameters,
        )


__all__ = ['SyntheticParameters']
