from functools import partial
from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field

from acoupipe.new_datasets.models.environments import BasePropagationModel
from acoupipe.new_datasets.models.signals import BaseSignalModel


def _create_source_signals(
    data: Dict[str, Any], env_func: Callable, fs: int, dtype: str
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
    loc = data["loc"]
    signals = data["source_signals"]
    if "noisy_mic_pos" not in data:
        mic_pos = data["mic_pos"]
    else:
        mic_pos = data["noisy_mic_pos"]
    assert loc.shape[1] == signals.shape[1], "Number of sources and signals must match."
    mic_signals = np.empty(signals.shape + (mic_pos.shape[1],), dtype=dtype)

    ts = ac.TimeSamples(sample_freq=fs)
    sig = ac.GenericSignalGenerator(source=ts)
    ps = ac.PointSource(
        signal=sig, mics=ac.MicGeom(pos_total=mic_pos), env=env_func(data))
    for i in range(signals.shape[1]):
        ts.data = signals[:,i][:,np.newaxis]
        ps.loc = tuple(loc[:, i])
        mic_signals[:, i] = ac.tools.return_result(ps)
    return mic_signals

class BaseSourceModel(BaseModel):
    """Base class for all noise models."""

    model_type: str = Field(..., description="Type of noise signals to generate (e.g., 'uncorrelated-wnoise').")
    signal_model: BaseSignalModel
    propagation_model: BasePropagationModel

    def create_mic_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create signals."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MonopoleSourceModel(BaseSourceModel):
    model_type: Literal["monopole"] = "monopole"

    def create_mic_signals_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        propagation_model_fn = self.propagation_model.create_env_fn()
        return partial(
            _create_source_signals,
            env_func=propagation_model_fn,
            fs=self.signal_model.fs,
            dtype=self.signal_model.dtype,
        )


SOURCE_MODEL_MAPPING = {
    "monopole": MonopoleSourceModel
}

class SourceModel(BaseModel):
    model_type: str = Field(..., description="Type of source to generate (e.g., 'monopole').")

    @staticmethod
    def configure_model(**kwargs: Dict[str, Any]) -> BaseSourceModel:
        model_type = kwargs["model_type"]
        model_class = SOURCE_MODEL_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported source type: {model_type}")
        return model_class(**kwargs)


# Example Usage
if __name__ == "__main__":
    from acoupipe.new_datasets.models.environments import PropagationModel
    from acoupipe.new_datasets.models.signals import SignalModel
    # Input data for the factory
    input_data = {
        "model_type": "monopole",
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
    prop_model = PropagationModel.configure_model(model_type="free-field", c=440)

    # Create a specific signal model using the factory
    src_model = SourceModel.configure_model(
        signal_model=signal_model, propagation_model=prop_model, **input_data)

    # Get the signal generation function
    fn = src_model.create_mic_signals_fn()

    # Generate signals
    signals = fn({
        "mic_pos": np.ones((3, 10)),
        "loc": np.random.randn(3*5).reshape((3, 5)),
        "signals": np.random.rand(10000,5),
    })
    print("Generated Signals Shape:", signals.shape)

