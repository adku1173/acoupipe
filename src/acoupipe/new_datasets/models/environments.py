from typing import Any, Callable, Dict, Literal

import acoular as ac
import numpy as np
from pydantic import BaseModel, Field

from acoupipe.new_datasets.models.base import BaseModelSubConfig
from acoupipe.new_datasets.transfer import TransferMonopole


def _create_free_field_env(data: Dict[str, Any]) -> ac.Environment:
    return ac.Environment(c=data["c"])

def _create_free_field_transfer(data: Dict[str, Any]) -> TransferMonopole:
    """
    Create a transfer function for free field propagation.

    Args:
        data (dict): Input data dictionary.
        c (float): Speed of sound.

    Returns
    -------
        TransferMonopole: Transfer function object.
    """
    mic_pos_total = data.get("noisy_mic_pos")
    if mic_pos_total is None:
        mic_pos_total = data["mic_pos"]
    return TransferMonopole(ref=data["ref"], env=_create_free_field_env(data),
        mics=ac.MicGeom(pos_total=mic_pos_total), grid=ac.ImportGrid(pos=data["loc"]))


class BasePropagationModel(BaseModelSubConfig):
    """Base class for all signal models."""

    model_type: str = Field(..., description="Type of propagation model to use (e.g., 'free-field').")
    ref: list = Field(..., description="Reference point for the transfer function.")

    def create_env_fn(self) -> Callable[[Dict[str, Any]], np.ndarray]:
        """Get the function to create environment."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class FreeField(BasePropagationModel):
    """Model for generating free field environment."""

    model_type: Literal["free-field"] = "free-field"
    c: float = Field(default=343.0, description="The number of source signals.")

    def create_env_fn(self):
        """Get the function to create white noise signals."""
        return _create_free_field_env

    def create_transfer_fn(self):
        """Get the function to create transfer function."""
        return _create_free_field_transfer


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
