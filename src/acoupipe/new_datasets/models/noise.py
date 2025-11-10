from functools import partial
from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field

from acoupipe.new_datasets.models.base import BaseModelSubConfig


def _create_wnoise_signals(data: Dict[str, Any], dtype: str) -> np.ndarray:
    """
    Generate white noise signals.

    Args:
        data (dict): Input data dictionary.
        signal_length (int): Length of the signal in seconds.
        fs (int): Sampling frequency in Hz.
        dtype (str): Data type of the generated signals.

    Returns
    -------
        np.ndarray: Generated signals.
    """
    fs = data["fs"]
    rms = data["rms"]
    nsam = int(data["signal_length"]*fs)
    seed = data["signal_seeds"] + 1000
    rms = (rms**2).sum() * data["noise_variance"]

    signal = ac.WNoiseGenerator(sample_freq=fs, num_samples=nsam, rms=rms)
    noise_source = ac.UncorrelatedNoiseSource(
            signal=signal,
            seed=seed,
            mics=ac.MicGeom(pos_total=data["mic_pos"]),
        )
    signals = np.empty((noise_source.num_channels, nsam), dtype=dtype)
    signals[:,:] = ac.tools.return_result(noise_source).T
    return signals

class BaseNoiseModel(BaseModelSubConfig):
    """Base class for all noise models."""

    model_type: str = Field(..., description="Type of noise signals to generate (e.g., 'uncorrelated-wnoise').")
    precision: Literal["single", "double"] = Field(
        default="single", description="Precision of the generated signals."
    )

    def create_signal_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create signals."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class UncorrelatedWNoiseModel(BaseNoiseModel):
    """Model for generating white noise signals."""

    # TODO: noise variance is not the correct word here
    model_type: Literal["uncorrelated-wnoise"] = "uncorrelated-wnoise"
    noise_variance: float = Field(default=1.0, description="Variance of the noise.")

    def create_signal_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        return partial(_create_wnoise_signals, dtype="float32" if self.precision == "single" else "float64")

    def create_noise_csm_fn(self):
        def _create_noise_csm(
                data: Dict[str, Any], dtype="complex64") -> np.ndarray:
            """Create a cross-spectral matrix (CSM) of white noise signals for the desired frequencies."""
            nmics = data["mic_pos"].shape[-1]
            nfft = data["nfft"]
            nfreq = len(data["f_indices"])
            rms = (data["rms"]**2).sum() / nfft * data["noise_variance"]
            q_matrix = np.zeros((nfreq, nmics, nmics), dtype=dtype)
            for n in range(nfreq):
                q_matrix[n, :, :] = np.diag(np.ones(nmics) * rms)
            return q_matrix
        return partial(_create_noise_csm, dtype="complex64" if self.precision == "single" else "complex128")


NOISE_MODEL_MAPPING = {
    "uncorrelated-wnoise": UncorrelatedWNoiseModel
}

class NoiseModel(BaseModel):
    model_type: str = Field(..., description="Type of signal to generate (e.g., 'wnoise').")

    @staticmethod
    def configure_model(**kwargs: Dict[str, Any]) -> BaseNoiseModel:
        model_type = kwargs["model_type"]
        model_class = NOISE_MODEL_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported signal type: {model_type}")
        return model_class(**kwargs)


# Example Usage
if __name__ == "__main__":
    from acoupipe.new_datasets.models.signals import SignalModel
    from acoupipe.new_datasets.models.spectra import FrequencyModel

    freq_model = FrequencyModel


    # Input data for the factory
    input_data = {
        "model_type": "uncorrelated-wnoise",
    }

    sig_input_data = {
        "model_type": "wnoise",
        "nsources": 3,
        "signal_length": 5,
        "dtype": "float32",
    }

    # Create a specific signal model using the factory
    signal_model = SignalModel.configure_model(**sig_input_data)

    # Create a specific signal model using the factory
    noise_model = NoiseModel.configure_model(signal_model=signal_model, **input_data)

    # Get the signal generation function
    fn = noise_model.create_signal_fn()

    # Generate signals
    signals = fn({"mic_pos": np.ones((3, 10))})
    print("Generated Signals Shape:", signals.shape)

