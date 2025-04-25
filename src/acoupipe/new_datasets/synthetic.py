from abc import abstractmethod
from typing import Optional

from acoupipe.new_datasets.config import DatasetConfig
from acoupipe.new_datasets.datasource import range as ac_range
from acoupipe.new_datasets.features import (
    FeatureCollectionBuilder,
    MicSourceSignalFeature,
    SourceSignalFeature,
    TimeDataFeature,
)
from acoupipe.new_datasets.models.environments import PropagationModel
from acoupipe.new_datasets.models.signals import SignalModel
from acoupipe.new_datasets.models.sources import SourceModel
from acoupipe.new_datasets.monte_carlo import MonteCarloFactory


def get_numba_threads():
    """Get the number of threads for Numba."""
    import numba
    return numba.get_num_threads()

class DatasetBase:
    """Base class for datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def get_ray_dataset(self, features=None, size=None, start_idx=0, split=None, tasks=None, **kwargs):
        """Generate a Ray dataset."""
        feature_names = features
        size = size or self.config.dataset_size
        split = split or self.config.split

        # if size < 1e6:
        #     ds = ray.data.from_items(
        #         [{"idx": i} for i in range(start_idx, start_idx + size)])
        # else:
        # Create the base dataset
        ds = ac_range(
            start=start_idx, stop=start_idx + size, name=split, override_num_blocks=tasks)

        # Monte-Carlo
        #ds = self.monte_carlo.add_ray_map(ds, split=split, **kwargs)

        # Add features
        features = self.get_features()
        feature_builder = FeatureCollectionBuilder(*features)

        mc_fn = self.monte_carlo.build_map_fn(split=split)
        feat_fn = feature_builder.build_feature_func()

        # limit the number of threads to 1, gouverned by the map_fn

        def map_fn(data):
            mc_fn(data)
            feat_fn(data)
            # only return the feature_name items
            if feature_names is not None:
                out_data = {k: data[k] for k in feature_names}
            else:
                out_data = {}
            # # meta data (get numba num threads)
            # out_data["meta"] = {
            #     "num_threads": get_numba_threads(),
            # }
            return out_data

        ds = ds.map(map_fn, runtime_env={"env_vars": {"NUMBA_NUM_THREADS": "1"}}, **kwargs)
        return ds

    @abstractmethod
    def get_features(self):
        """Return the list of features for the dataset."""
        return []

    def to_json(self):
        """Serialize the dataset configuration to JSON."""
        return self.config.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_data: str):
        """Deserialize the dataset configuration from JSON."""
        config = DatasetConfig.model_validate_json(json_data)
        return cls(config)


class DatasetSynthetic(DatasetBase):
    """Synthetic dataset class."""

    def __init__(self, config: Optional[DatasetConfig] = None, **kwargs):
        # Use default configuration if none is provided
        if config is None:
            config = DatasetConfig().configure(**kwargs)
        super().__init__(config)
        self.monte_carlo = MonteCarloFactory.configure_model(**self.config.monte_carlo)
        self.signal_model = SignalModel.configure_model(**self.config.signal_model)
        self.propagation_model = PropagationModel.configure_model(**self.config.propagation_model)
        self.source_model = SourceModel.configure_model(signal_model=self.signal_model,
                            propagation_model=self.propagation_model, **self.config.source_model)

    def get_features(self):
        """Return the features for the synthetic dataset."""
        return [
           SourceSignalFeature(signal_model=self.signal_model, name="source_signals"),
           MicSourceSignalFeature(source_model=self.source_model),
           TimeDataFeature(),
        ]


if __name__ == "__main__":
    import numpy as np

    dataset = DatasetSynthetic(
        fs=44100, mic_pos=np.random.randn(3*10).reshape((3,10)).tolist())

    ray_ds = dataset.get_ray_dataset(size=500, concurrency=1)
    res = ray_ds.take(1)
    for d in res:
        print(d["source_signals"].shape)


    # Serialize the configuration to JSON
    json_config = dataset.to_json()
    print("\nDataset Configuration as JSON:")
    print(json_config)

    # Deserialize the configuration from JSON
    new_dataset = DatasetSynthetic.from_json(json_config)
    print("\nDeserialized Dataset Configuration:")
