from abc import abstractmethod


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
        fn = self.signal_model.create_signals_fn()
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
