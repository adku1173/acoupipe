"""Synthetic ISM dataset parameters."""

from acoupipe.datasets.synthetic.parameters import SyntheticParameters


class SyntheticISMParameters(SyntheticParameters):
    """
    Shallow simulation parameters for synthetic image-source-method datasets.

    This schema extends :class:`SyntheticParameters` with room acoustics state
    only. Processing-engine settings and Feature-specific parameters remain out
    of scope.
    """

    def __init__(self, rt60=2.0, room_size=None, **parameters):
        if room_size is None:
            room_size = [6, 4, 3]
        super().__init__(**parameters)
        self.add_parameter('rt60', rt60)
        self.add_parameter('room_size', room_size)


__all__ = ['SyntheticISMParameters']
