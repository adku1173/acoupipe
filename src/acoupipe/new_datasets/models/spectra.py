from typing import Literal, Union
from warnings import warn

import numpy as np
from pydantic import BaseModel, Field, computed_field, field_validator

from acoupipe.new_datasets.models.base import BaseModelSubConfig


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
        ind = np.searchsorted(freq, f)
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
        ind1 = np.searchsorted(freq, f1)
        ind2 = np.searchsorted(freq, f2)
        if ind1 == ind2:
            warn("Queried frequency band (%g to %g Hz) does not "
                    "include any discrete FFT sample frequencies. "
                    "Returning zeros." % (f1,f2),
                    Warning, stacklevel = 2)
        return (ind1,ind2)



class FFTModel(BaseModelSubConfig):

    model_type: Literal["fft"] = "fft"
    fs: int = Field(default=13720, description="Sampling frequency in Hz.")
    block_size: int = Field(default=128, description="Block size for FFT.")
    overlap: str = Field(default="50%", description="Overlap percentage for FFT.")
    num: int = Field(default=0, description="Frequency band width. Default is 0 (single frequency line).")
    f: list[float] = Field(default_factory=list, repr=False)

    @property
    def overlap_(self) -> int:
        """Convert overlap percentage to integer."""
        overlap_mapping = {"None": 1, "50%": 2, "75%": 4, "87.5%": 8}
        if self.overlap not in overlap_mapping:
            raise ValueError(f"Invalid overlap value: {self.overlap}. Must be one of {list(overlap_mapping.keys())}.")
        return overlap_mapping[self.overlap]

    def model_post_init(self, __context):
        # Set f to fftfreq if f is empty
        if not self.f:
            self.f = list(self.fftfreq)[1:]

    @field_validator("f", mode="before")
    @classmethod
    def validate_f(cls, value: Union[float, int, list[float]]) -> list[float]:
        if isinstance(value, (float, int)):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError("The `f` attribute must be a float, int, or list of floats.")

    @computed_field
    @property
    def f_indices(self) -> list:
        """Return the left and right indices that define the frequency range to integrate over."""
        indices = []
        for indices_tuple in self.f_indices_tuple_list:
            indices += list(range(indices_tuple[0], indices_tuple[1]))
        indices = list(set(indices))
        return indices

    @computed_field
    @property
    def f_indices_tuple_list(self) -> list:
        """Return the left and right indices that define the frequency range to integrate over."""
        inds = []
        for f_ in self.f:
            inds.append(
                get_frequency_index_range(self.fftfreq, f_, self.num)
            )
        return inds

    @computed_field
    @property
    def nfft(self) -> int:
        """Calculate nfft based on block_size."""
        return self.block_size // 2 + 1

    @computed_field
    @property
    def fftfreq(self) -> list:
        """Return the Discrete Fourier Transform sample frequencies."""
        return abs(np.fft.fftfreq(self.block_size, 1.0 / self.fs)[: int(self.block_size / 2 + 1)])


FREQ_MODEL_MAPPING = {
    "fft": FFTModel,
}

class FrequencyModel(BaseModel):
    """Factory class for creating specific frequency models."""

    model_type: str = Field(..., description="Type of frequency model to generate (e.g., 'base').")

    @staticmethod
    def configure_model(**kwargs):
        """Create a specific frequency model based on the model type."""
        model_type = kwargs["model_type"]
        model_class = FREQ_MODEL_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported frequency type: {model_type}")
        return model_class(**kwargs)


if __name__ == "__main__":

    # Example Usage
    input_data = {
        "model_type": "fft",
        "fs": 13720,
        "block_size": 1024,
        "overlap": "50%",
        "num": 3,
        "f": [1000, 2000, 3000],
    }

    frequency_model = FrequencyModel.configure_model(**input_data)
    print(frequency_model.model_dump().keys())
    print(frequency_model)
    print(frequency_model.f_indices)
