from functools import partial
from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field

from acoupipe.new_datasets.models.signals import BaseSignalModel


def _create_wnoise_signals(
    data: Dict[str, Any], signal_length: float, fs: int, dtype: str
) -> np.ndarray:
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
    nsam = data.get("signal_length", signal_length) * fs
    fs = data.get("fs", fs)
    rms = data.get("rms", np.ones((1,)))  # Default RMS value if not provided
    noise_variance = data.get("noise_variance", 1.0)  # Default noise variance if not provided
    rms = (rms**2).sum() * noise_variance  # Use noise_variance instead of data["noise_variance"]
    nsrc = data["mic_pos"].shape[1]
    seed = data.get("signal_seeds", np.arange(nsrc)) + 1000
    signal = ac.WNoiseGenerator(sample_freq=fs, num_samples=nsam, rms=rms)
    noise_source = ac.UncorrelatedNoiseSource(
            signal=signal,
            seed=seed,
            mics=ac.MicGeom(pos_total=data["mic_pos"]),
        )
    signals = np.empty((noise_source.num_channels, nsam), dtype=dtype)
    signals[:,:] = ac.tools.return_result(noise_source).T
    return signals

class BaseNoiseModel(BaseModel):
    """Base class for all noise models."""

    model_type: str = Field(..., description="Type of noise signals to generate (e.g., 'uncorrelated-wnoise').")
    signal_model: BaseSignalModel

    def create_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create signals."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class UncorrelatedWNoiseModel(BaseNoiseModel):
    """Model for generating white noise signals."""

    model_type: Literal["uncorrelated-wnoise"] = "uncorrelated-wnoise"

    def create_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        return partial(
            _create_wnoise_signals,
            signal_length=self.signal_model.signal_length,
            fs=self.signal_model.fs,
            dtype=self.signal_model.dtype,
        )


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

    # Input data for the factory
    input_data = {
        "model_type": "uncorrelated-wnoise",
    }

    sig_input_data = {
        "model_type": "wnoise",
        "nsources": 3,
        "signal_length": 5,
        "fs": 44100,
        "dtype": "float32",
    }

    # Create a specific signal model using the factory
    signal_model = SignalModel.configure_model(**sig_input_data)

    # Create a specific signal model using the factory
    noise_model = NoiseModel.configure_model(signal_model=signal_model, **input_data)

    # Get the signal generation function
    fn = noise_model.create_signals_fn()

    # Generate signals
    signals = fn({"mic_pos": np.ones((3, 10))})
    print("Generated Signals Shape:", signals.shape)

