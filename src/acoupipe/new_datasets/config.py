from pathlib import Path
from typing import Any, Dict

import acoular as ac
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MICS = ac.MicGeom(
            file=Path(ac.__file__).parent / "xml" / "tub_vogel64.xml")
DEFAULT_MICS.pos_total /= DEFAULT_MICS.aperture


# def serialize_callable(func):
#     return {"__callable__": f"{func.__module__}.{func.__name__}"}

# def deserialize_callable(obj):
#     if isinstance(obj, dict) and "__callable__" in obj:
#         module_name, func_name = obj["__callable__"].rsplit(".", 1)
#         module = importlib.import_module(module_name)
#         return getattr(module, func_name)
#     return obj



class DatasetConfig(BaseModel):
    """Configuration schema for the dataset."""

    model_config = ConfigDict(extra="allow")

    monte_carlo: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model_type": "synthetic", "mic_pos_noise": True, "mic_pos": DEFAULT_MICS.pos_total.tolist(),
            "mic_pos_noise_ddir": np.array([[1.0], [1.0], [0]]).tolist(), "signal_length": 5},
        description="Configuration for the Monte Carlo model."
    )

    signal_model: Dict[str, Any] = Field(
        default_factory=lambda: {"model_type": "wnoise", "fs": 13720, "signal_length": 5},
        description="Configuration for the signal model."
    )

    propagation_model: Dict[str, Any] = Field(
        default_factory=lambda: {"model_type": "free-field", "c": 343.0},
        description="Configuration for the propagation model."
    )

    source_model: Dict[str, Any] = Field(
        default_factory=lambda: {"model_type": "monopole"},
        description="Configuration for the source model."
    )

    dataset_size: int = Field(
        default=1000,
        description="Number of samples in the dataset."
    )
    split: str = Field(
        default="training",
        description="Dataset split type (e.g., 'training', 'validation', 'test')."
    )

    def configure(self, **overrides) -> "DatasetConfig":
        """
        Create a new configuration with specific overrides.

        Args:
            **overrides: Key-value pairs of parameters to override.

        Returns
        -------
            DatasetConfig: A new configuration with the overrides applied.
        """
        updated_config = self.model_dump()
        for key, value in overrides.items():
            if key in updated_config:
                updated_config[key] = value
            elif key in updated_config.get("monte_carlo", {}):
                updated_config["monte_carlo"][key] = value
            elif key in updated_config.get("signal_model", {}):
                updated_config["signal_model"][key] = value
            elif key in updated_config.get("propagation_model", {}):
                updated_config["propagation_model"][key] = value
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        return DatasetConfig(**updated_config)

    # TODO: implement custom serialization and deserialization functions

    # def model_dump(self, *args, **kwargs):
    #     raw = super().model_dump(*args, **kwargs)
    #     raw["monte_carlo"] = self._serialize_nested_dict(raw["monte_carlo"])
    #     return raw

    # @staticmethod
    # def _serialize_nested_dict(d):
    #     def recurse(obj):
    #         if callable(obj):
    #             return serialize_callable(obj)
    #         elif isinstance(obj, dict):
    #             return {k: recurse(v) for k, v in obj.items()}
    #         else:
    #             return obj
    #     return recurse(d)

    # @classmethod
    # def model_validate_json(cls, json_data):
    #     import json
    #     data = json.loads(json_data)
    #     data["monte_carlo"] = cls._deserialize_nested_dict(data["monte_carlo"])
    #     return cls(**data)

    # @staticmethod
    # def _deserialize_nested_dict(d):
    #     def recurse(obj):
    #         if isinstance(obj, dict):
    #             if "__callable__" in obj:
    #                 return deserialize_callable(obj)
    #             return {k: recurse(v) for k, v in obj.items()}
    #         else:
    #             return obj
    #     return recurse(d)
