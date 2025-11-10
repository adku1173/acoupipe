from typing import Any, Callable, Dict

from pydantic import BaseModel


class BaseModelSubConfig(BaseModel):

    def get_expose_fn(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the function to expose the model parameters.

        Returns
        -------
            Callable: Function to expose the model parameters.
        """
        model_data = self.model_dump()

        def expose_to_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Expose the signal model to the data dictionary.

            Args:
                data (dict): Input data dictionary.

            Returns
            -------
                dict: Updated data dictionary with signal model parameters.
            """
            # update only items which are not already in data
            for key, value in model_data.items():
                if key in ["precision", "dtype"]:
                    continue
                if key not in data:
                    data[key] = value
            return data
        return expose_to_data
