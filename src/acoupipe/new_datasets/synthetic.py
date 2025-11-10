from abc import abstractmethod
from typing import Optional

from acoupipe.new_datasets.config import DatasetConfig
from acoupipe.new_datasets.datasource import range as ac_range
from acoupipe.new_datasets.features import CSMFeature, FeatureCollectionBuilder, MicCSMFeature, MicNoiseFeature, SourceCSMFeature
from acoupipe.new_datasets.models.environments import PropagationModel
from acoupipe.new_datasets.models.noise import NoiseModel
from acoupipe.new_datasets.models.signals import SignalModel
from acoupipe.new_datasets.models.sources import SourceModel
from acoupipe.new_datasets.models.spectra import FrequencyModel
from acoupipe.new_datasets.monte_carlo import MonteCarloFactory


def get_numba_threads():
    """Get the number of threads for Numba."""
    import numba
    return numba.get_num_threads()

class DatasetBase:
    """Base class for datasets."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sub_config = []

    def get_ray_dataset(self, features, size=None, start_idx=0, split=None, tasks=None, **kwargs):
        """Generate a Ray dataset."""
        size = size or self.config.dataset_size
        split = split or self.config.split
        map_fn = self._build_map_fn(feature_names=features, split=split)
        # Create the base dataset
        ds = ac_range(
            start=start_idx, stop=start_idx + size, name=split, override_num_blocks=tasks)

        ds = ds.map(map_fn, runtime_env={"env_vars": {"NUMBA_NUM_THREADS": "1"}}, **kwargs)
        return ds

    def _build_map_fn(self, feature_names=False, split=None):

        # Add features
        features = self.get_features()
        feature_builder = FeatureCollectionBuilder(*features)

        split = split or self.config.split #TODO: why config.split here?
        mc_fn = self.monte_carlo.build_map_fn(split=split)
        expose_config_fn = self._build_expose_config_fn()
        feat_fn = feature_builder.build_feature_func()

        def map_fn(data):
            mc_fn(data)
            expose_config_fn(data)
            feat_fn(data)
            # only return the feature_name items
            if feature_names:
                out_data = {k: data[k] for k in feature_names}
            else:
                out_data = data
            # # meta data (get numba num threads)
            # out_data["meta"] = {
            #     "num_threads": get_numba_threads(),
            # }
            return out_data
        return map_fn

    def _build_expose_config_fn(self):
        """Expose the configuration for the dataset."""
        fns = []
        for cf in self.sub_config:
            fns.append(cf.get_expose_fn())
        def expose_config_fn(data):
            for f in fns:
                data = f(data)
        return expose_config_fn

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
        self.frequency_model = FrequencyModel.configure_model(**self.config.frequency_model)
        self.propagation_model = PropagationModel.configure_model(**self.config.propagation_model)
        self.signal_model = SignalModel.configure_model(
            frequency_model=self.frequency_model, **self.config.signal_model)
        self.noise_model = NoiseModel.configure_model(**self.config.noise_model)
        self.source_model = SourceModel.configure_model(signal_model=self.signal_model,
            propagation_model=self.propagation_model, **self.config.source_model)

        self.sub_config = [
            self.frequency_model,
            self.propagation_model,
            self.signal_model,
            self.noise_model,
            self.source_model,
        ]

    def get_features(self):
        """Return the features for the synthetic dataset."""
        return [
           # CSM calculation
           SourceCSMFeature(signal_model=self.signal_model, mode="analytic", name="source_csm_analytic"),
           SourceCSMFeature(signal_model=self.signal_model, mode="wishart", name="source_csm_wishart"),
           MicCSMFeature(propagation_model=self.propagation_model, name="mic_csm_analytic", depends_on=["source_csm_analytic"]),
           MicCSMFeature(propagation_model=self.propagation_model, name="mic_csm_wishart", depends_on=["source_csm_wishart"]),
           MicNoiseFeature(noise_model=self.noise_model, mode="analytic", name="mic_noise_analytic"),
           MicNoiseFeature(noise_model=self.noise_model, mode="wishart", name="mic_noise_wishart"),
           CSMFeature(name="csm_analytic", depends_on=["mic_csm_analytic", "mic_noise_analytic"]),
           CSMFeature(name="csm_wishart", depends_on=["mic_csm_wishart", "mic_noise_wishart"]),
        #    SourceSignalFeature(signal_model=self.signal_model, name="source_signals"),
        #    MicSourceSignalFeature(source_model=self.source_model),
        #    TimeDataFeature(),

        #    MicCSMFeature(
        #        signal_model=self.signal_model,
        #        propagation_model=self.propagation_model,
        #        frequency_model=self.frequency_model,
        #    ),

        ]


if __name__ == "__main__":
    import numpy as np

    dataset = DatasetSynthetic(
        fs=44100, mic_pos=np.random.randn(3*10).reshape((3,10)).tolist())

    ray_ds = dataset.get_ray_dataset(size=5, features=["source_csm"])
    for d in ray_ds.iter_rows():
        print(d["source_csm"].shape)

    # Serialize the configuration to JSON
    json_config = dataset.to_json()
    print("\nDataset Configuration as JSON:")
    print(json_config)

    # Deserialize the configuration from JSON
    new_dataset = DatasetSynthetic.from_json(json_config)
    print("\nDeserialized Dataset Configuration:")
