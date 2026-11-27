"""The AcouPipe module extends the computational pipeline-based concept of Acoular_.

It provides additional tools that can be helpful to generate realizations of features
in a predefined random process.
"""

# Acoular configures OpenBLAS before NumPy is loaded (acoular/acoular#322).
from acoular import config as _acoular_config  # noqa: F401

from .version import __version__ as __version__

__author__ = 'Adam Kujawski, Art R. J. Pelling, Simon Jekosch, Ennes Sarradj'
