from abc import abstractmethod
from functools import partial

import numpy as np  # noqa: F401

from acoupipe.new_datasets.utils import get_overlap_ratio
from acoupipe.new_datasets.wishart import sample_wishart


class BaseFeature:
    """Base class defining the virtual feature."""

    def __init__(self, name, dtype, depends_on=None):
        self.name = name
        self.dtype = dtype
        if depends_on is not None:
            self.depends_on = depends_on
        else:
            self.depends_on = []

    @abstractmethod
    def build_feature_func(self):
        # build the feature function for the dataset
        # this is a stateless function that can be used in the map
        # function of the dataset.
        def map_fn(data):
            # add some features to the data
            return data
        return map_fn


class SourceSignalFeature(BaseFeature):
    """Feature class defining the source signal feature."""

    def __init__(
            self, signal_model, name="source_signals", dtype="float32", depends_on=None):
        """
        Initialize the SourceSignalFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
            signal_model (BaseSignalModel): The signal model type
        """
        super().__init__(name, dtype, depends_on)
        self.signal_model = signal_model

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        fn = self.signal_model.create_signal_fn()
        def feature_func(data):
            data[self.name] = fn(data)
            return data
        return feature_func


class MicSourceSignalFeature(BaseFeature):

    def __init__(
            self, source_model, name="mic_src_signals", dtype="float32", depends_on=["source_signals"]):
        super().__init__(name, dtype, depends_on)
        self.source_model = source_model

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        fn = self.source_model.create_mic_signals_fn()
        def feature_func(data):
            data[self.name] = fn(data)
            return data
        return feature_func


class TimeDataFeature(BaseFeature):
    """Feature class defining the microphone signal feature."""

    def __init__(
            self, name="time_data", dtype="float32", depends_on=["mic_src_signals"]):
        """
        Initialize the MicSignalFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
            propagation_model (BasePropagationModel): The propagation model type
        """
        super().__init__(name, dtype, depends_on)

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        def feature_func(data):
            data[self.name] = data["mic_src_signals"].sum(1)
            return data
        return feature_func

class SourceCSMFeature(BaseFeature):
    #TODO: there is something nasty with the dtype handling here
    def __init__(
            self, signal_model, mode, name="source_csm", dtype="float32", depends_on=None):
        """
        Initialize the SourceSignalFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
            signal_model (BaseSignalModel): The signal model type
        """
        super().__init__(name, dtype, depends_on)
        self.signal_model = signal_model
        self.mode = mode

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        fn = self.signal_model.create_signal_csm_fn()
        def feature_func(data, mode, name, dtype):
            q_matrix = fn(data)
            if mode == "wishart":
                indices = data["f_indices"]
                block_size = data["block_size"]
                num_samples = int(data["signal_length"] * data["fs"])
                overlap_ = get_overlap_ratio(data["overlap"])
                df = overlap_ * num_samples / block_size - overlap_ + 1
                for i,j in enumerate(indices):
                    seed = data["signal_seeds"][0]
                    rng = np.random.default_rng([seed, j])
                    q_matrix[i] = sample_wishart(q_matrix[i], df, rng)
            data[f"{name}"] = q_matrix
            return data
        return partial(feature_func, mode=self.mode, name=self.name, dtype=self.dtype)



class MicCSMFeature(BaseFeature):

    def __init__(self, propagation_model, name, depends_on, dtype="float32"):
        """
        Initialize the MicCSMFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
            propagation_model (BasePropagationModel): The propagation model type
        """
        super().__init__(name, dtype, depends_on)
        self.propagation_model = propagation_model

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        tf_fn = self.propagation_model.create_transfer_fn()
        def feature_func(data, q_name):
            Q = data[q_name]
            H = tf_fn(data).transfer(data["f"])
            csm = (H.swapaxes(2, 1) @ Q @ H.conjugate())
            data[f"{self.name}"] = csm
            return data
        return partial(feature_func, q_name=self.depends_on[0])

class MicNoiseFeature(BaseFeature):
    def __init__(self, noise_model, mode, name="mic_noise", dtype="float32", depends_on=None):
        """
        Initialize the MicNoiseFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
            noise_model (BaseNoiseModel): The noise model type
        """
        super().__init__(name, dtype, depends_on)
        self.noise_model = noise_model
        self.mode = mode

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        fn = self.noise_model.create_noise_csm_fn()
        def feature_func(data, mode, name, dtype):
            n_matrix = fn(data)
            if mode == "wishart":
                indices = data["f_indices"]
                block_size = data["block_size"]
                num_samples = int(data["signal_length"] * data["fs"])
                overlap_ = get_overlap_ratio(data["overlap"])
                df = overlap_ * num_samples / block_size - overlap_ + 1
                for i,j in enumerate(indices):
                    seed = data["signal_seeds"][0] + 1
                    rng = np.random.default_rng([seed, j])
                    n_matrix[i] = sample_wishart(n_matrix[i], df, rng)
            data[f"{name}"] = n_matrix
            return data
        return partial(feature_func, mode=self.mode, name=self.name, dtype=self.dtype)

class CSMFeature(BaseFeature):
    def __init__(self, name, depends_on, dtype="float32"):
        """
        Initialize the MicCSMFeature.

        Args:
            name (str): Name of the feature.
            dtype (str): Data type of the feature.
            depends_on (list): List of dependencies for the feature.
        """
        super().__init__(name, dtype, depends_on)

    def build_feature_func(self):
        """
        Build the feature function for the dataset.

        Returns
        -------
            Callable: Feature function to generate signals.
        """
        def feature_func(data, name, depends_on):
            data[name] = data[depends_on[0]] + data[depends_on[1]]
            return data
        return partial(feature_func, name=self.name, depends_on=self.depends_on)



class FeatureCollectionBuilder:

    def __init__(self, *features):
        self.features = features # the feature objects

    def topological_sort(self):
        # Build a dependency graph
        # Graph-Based Resolution: Model the feature dependencies as a directed acyclic
        # graph (DAG) and perform a topological sort.
        graph = {feat.name: set(feat.depends_on) for feat in self.features}
        sorted_features = []
        while graph:
            # Find features with no unmet dependencies
            ready = [name for name, deps in graph.items() if not deps]
            if not ready:
                raise ValueError("Circular dependency detected among features.")
            for name in ready:
                sorted_features.append(name)
                graph.pop(name)
            # Remove satisfied dependencies
            for deps in graph.values():
                deps.difference_update(ready)
        return sorted_features

    def build_feature_func(self):
        # build the feature function for the dataset only for the features that are
        # requested. This is a stateless function that can be used in the map
        # function of the dataset.
        feature_dict = {feat.name: feat for feat in self.features}
        feature_funcs = self.topological_sort()

        # build the feature functions
        built_feature_funcs = []
        for ff in feature_funcs:
            built_feature_funcs.append(
                feature_dict[ff].build_feature_func()
                )

        def feature_fn(data):
            for fn in built_feature_funcs:
                # TODO: maybe better to use map for each feature? (passing data around might be slow for large data)
                # but what if we want to use a stateful function in between? E.g. a model creating
                # RIRs?
                data = fn(data)
            return data
        # return the feature function
        return feature_fn
