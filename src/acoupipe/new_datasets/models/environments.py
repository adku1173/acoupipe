from functools import partial
from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field


def _create_free_field_env(
    data: Dict[str, Any], c: float
) -> ac.Environment:
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
    c_value = data.get("c", c)
    return ac.Environment(c=c_value)


class BasePropagationModel(BaseModel):
    """Base class for all signal models."""

    model_type: str = Field(..., description="Type of propagation model to use (e.g., 'free-field').")

    def create_env_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class FreeField(BasePropagationModel):
    """Model for generating free field environment."""

    model_type: Literal["free-field"] = "free-field"
    c: float = Field(default=343.0, description="The number of source signals.")

    def create_env_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create white noise signals."""
        return partial(_create_free_field_env, c=self.c)


ENV_MODEL_MAPPING = {
    "free-field": FreeField
}

class PropagationModel(BaseModel):
    """Factory class for creating specific signal models."""

    model_type: str = Field(..., description="Type of signal to generate (e.g., 'wnoise').")

    @staticmethod
    def configure_model(**kwargs: Dict[str, Any]) -> BasePropagationModel:
        """Factory method to create a specific environment based on the model type."""
        model_type = kwargs["model_type"]
        model_class = ENV_MODEL_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported signal type: {model_type}")
        return model_class(**kwargs)


# Example Usage
if __name__ == "__main__":
    # Input data for the factory
    input_data = {
        "model_type": "free-field",
        "c": 400.0,
    }

    signal_model = PropagationModel.configure_model(**input_data)
    create_env_fn = signal_model.create_env_fn()

    # Generate signals
    env = create_env_fn(input_data)
    print("speed of sound:", env.c)
