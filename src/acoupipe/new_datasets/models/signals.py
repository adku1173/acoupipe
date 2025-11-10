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
    nsrc = data["nsources"]
    fs = data["fs"]
    rms = data["rms"]
    nsam = int(data["signal_length"]*fs)
    seeds = data["signal_seeds"]

    signals = np.empty((nsam, nsrc), dtype=dtype)
    for i in range(nsrc):
        signals[:, i] = ac.WNoiseGenerator(
            sample_freq=fs,
            num_samples=nsam,
            rms=rms[i],
            seed=seeds[i],
        ).signal()
    return signals


class BaseSignalModel(BaseModelSubConfig):
    """Base class for all signal models."""

    model_type: str = Field(..., description="Type of signal to generate (e.g., 'wnoise').")

    def create_signal_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create signals."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class WhiteNoiseSignalModel(BaseSignalModel):
    """Model for generating white noise signals."""

    model_type: Literal["wnoise"] = "wnoise"
    nsources: int = Field(default=1, description="The number of source signals.")
    rms: list = Field(default=[1.], description="RMS values for each source signal.")
    signal_seeds: list = Field(
        default=[1], description="Seeds for the random number generator for each source signal."
    )
    signal_length: float = Field(default=5, description="Length of the signal in seconds.")
    precision: Literal["single", "double"] = Field(
        default="single", description="Precision of the generated signals."
    )

    def create_signal_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        return partial(
            _create_wnoise_signals,
            dtype="float32" if self.precision == "single" else "float64",
        )

    def create_signal_csm_fn(self):
        def _create_signal_csm(
                data: Dict[str, Any], dtype="complex64") -> np.ndarray:
            """Create a cross-spectral matrix (CSM) of white noise signals for the desired frequencies."""
            nsources = data["nsources"]
            rms = data["rms"]
            nfft = data["nfft"]
            nfreq = len(data["f_indices"])
            rms = rms**2 / nfft
            q_matrix = np.zeros((nfreq, nsources, nsources), dtype=dtype)
            for n in range(nfreq):
                q_matrix[n, :, :] = np.diag(rms)
            return q_matrix
        return partial(_create_signal_csm, dtype="complex64" if self.precision == "single" else "complex128")


SIGNAL_MODEL_MAPPING = {
    "wnoise": WhiteNoiseSignalModel
}

class SignalModel(BaseModel):
    """Factory class for creating specific signal models."""

    model_type: str = Field(..., description="Type of signal to generate (e.g., 'wnoise').")

    @staticmethod
    def configure_model(**kwargs: Dict[str, Any]) -> BaseSignalModel:
        """Factory method to create a specific signal model based on the model type."""
        model_type = kwargs["model_type"]
        model_class = SIGNAL_MODEL_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported signal type: {model_type}")
        return model_class(**kwargs)

    # @classmethod
    # def from_json(cls, json_data: str) -> BaseSignalModel:
    #     """
    #     Deserialize JSON data and create the appropriate signal model.

    #     Args:
    #         json_data (str): JSON string representing the signal model.

    #     Returns
    #     -------
    #         BaseSignalModel: The deserialized signal model instance.
    #     """
    #     # Parse the JSON to determine the signal type
    #     data = cls.model_validate_json(json_data)
    #     model_type = data.model_type
    #     model_class = SIGNAL_MODEL_MAPPING.get(model_type)
    #     if not model_class:
    #         raise ValueError(f"Unsupported signal type: {model_type}")
    #     return model_class.model_validate_json(json_data)


# Example Usage
if __name__ == "__main__":
    # Input data for the factory
    input_data = {
        "model_type": "wnoise",
        "nsources": 3,
        "signal_length": 5,
        "fs": 44100,
        "dtype": "float32",
    }

    # Create a specific signal model using the factory
    signal_model = SignalModel.configure_model(**input_data)

    # Get the signal generation function
    create_signal_fn = signal_model.create_signal_fn()

    # Generate signals
    signals = create_signal_fn(input_data)
    print("Generated Signals Shape:", signals.shape)

    # # Serialize the specific signal model to JSON
    # json_data = signal_model.model_dump_json(indent=4)
    # print("\nSignal Model as JSON:")
    # print(json_data)

    # # Deserialize the model from JSON using the factory
    # loaded_model = SignalModel.from_json(json_data)
    # print("\nLoaded Signal Model from JSON:")
    # print(loaded_model)
