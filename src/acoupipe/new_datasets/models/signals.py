from functools import partial
from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field


def _create_wnoise_signals(
    data: Dict[str, Any], nsources: int, signal_length: float, fs: int, dtype: str
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
    nsrc = data.get("nsources", nsources)
    nsam = int(data.get("signal_length", signal_length) * fs)
    fs = data.get("fs", fs)
    rms = data.get("rms", np.ones(nsrc))  # Default RMS values if not provided
    seeds = data.get("signal_seeds", np.arange(nsrc))  # Default seeds if not provided
    signals = np.empty((nsam, nsrc), dtype=dtype)
    for i in range(nsrc):
        signals[:, i] = ac.WNoiseGenerator(
            sample_freq=fs,
            num_samples=nsam,
            rms=rms[i],
            seed=seeds[i],
        ).signal()
    return signals


class BaseSignalModel(BaseModel):
    """Base class for all signal models."""

    model_type: str = Field(..., description="Type of signal to generate (e.g., 'wnoise').")

    def create_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create signals."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class WhiteNoiseSignalModel(BaseSignalModel):
    """Model for generating white noise signals."""

    model_type: Literal["wnoise"] = "wnoise"
    nsources: int = Field(default=1, description="The number of source signals.")
    signal_length: float = Field(default=5, description="Length of the signal in seconds.")
    fs: int = Field(default=13720, description="Sampling frequency in Hz.")
    dtype: str = Field(default="float32", description="Data type of the generated signals.")

    def create_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        return partial(
            _create_wnoise_signals,
            nsources=self.nsources,
            signal_length=self.signal_length,
            fs=self.fs,
            dtype=self.dtype,
        )


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
    create_signals_fn = signal_model.create_signals_fn()

    # Generate signals
    signals = create_signals_fn(input_data)
    print("Generated Signals Shape:", signals.shape)

    # # Serialize the specific signal model to JSON
    # json_data = signal_model.model_dump_json(indent=4)
    # print("\nSignal Model as JSON:")
    # print(json_data)

    # # Deserialize the model from JSON using the factory
    # loaded_model = SignalModel.from_json(json_data)
    # print("\nLoaded Signal Model from JSON:")
    # print(loaded_model)
