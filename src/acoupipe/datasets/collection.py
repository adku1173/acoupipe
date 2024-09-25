
from functools import partial

from traits.api import Dict, HasPrivateTraits, Instance, List

from acoupipe.config import TF_FLAG


class BaseFeatureCollection(HasPrivateTraits):
    """
    BaseFeatureCollection base class for handling feature funcs.

    Attributes
    ----------
    feature_funcs : list
        List of feature_funcs.
    """

    feature_funcs = List(desc="list of feature_funcs")
    feature_tf_encoder_mapper = Dict(desc="feature encoder mapper")
    feature_tf_shape_mapper = Dict(desc="feature shape mapper")
    feature_tf_dtype_mapper = Dict(desc="feature dtype mapper")

    def add_feature_func(self, feature_func): #TODO: remove some unnecessary functions here!
        """
        Add a feature_func to the BaseFeatureCollection.

        Parameters
        ----------
        feature_func : str
            Feature to be added.
        """
        self.feature_funcs.append(feature_func)

    def get_feature_funcs(self):
        """
        Get all feature_funcs of the BaseFeatureCollection.

        Returns
        -------
        list
            List of feature_funcs.
        """
        def calc_features(sampler, feature_funcs):
            data = {}
            for ffunc in feature_funcs:
                data.update(ffunc(sampler=sampler))
            return data
        return partial(calc_features, feature_funcs=self.feature_funcs)

if TF_FLAG:
    import tensorflow as tf
    def get_output_signature(self, features):
        """Get the output signature of the dataset features.

        Parameters
        ----------
        features : list
            List of features included in the dataset. The features "seeds" and "idx" are always included.

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        dict
            Output signature of the dataset.
        """
        signature = {}
        for feature in features:
            if isinstance(feature, str):
                feature_name = feature
            else:
                feature_name = feature.name
            signature[feature_name] = tf.TensorSpec(
                self.feature_tf_shape_mapper[feature_name],self.feature_tf_dtype_mapper[feature_name])
        return signature
    BaseFeatureCollection.get_output_signature = get_output_signature


class FeatureCollectionBuilder(HasPrivateTraits):
    """
    FeatureCollectionBuilder base class for building a BaseFeatureCollection.

    Attributes
    ----------
    feature_collection : BaseFeatureCollection
        BaseFeatureCollection object.
    """

    feature_collection = Instance(BaseFeatureCollection, args=(), desc="BaseFeatureCollection object")

    def _add_tf_mapper(self, feature):
        if TF_FLAG:
            self.feature_collection.feature_tf_encoder_mapper.update(
                feature.get_tf_encoder_mapper())
            self.feature_collection.feature_tf_shape_mapper.update(
                feature.get_tf_shape_mapper())
            self.feature_collection.feature_tf_dtype_mapper.update(
                feature.get_tf_dtype_mapper())

    def add_custom(self, feature_func):
        """
        Add a custom feature to the BaseFeatureCollection.

        Parameters
        ----------
        feature_func : str
            Feature to be added.
        """
        self.feature_collection.add_feature_func(feature_func)

    def add_feature(self, feature):
        """Add an arbitrary feature to the BaseFeatureCollection.

        Parameters
        ----------
        feature : instance of :class:`~acoupipe.datasets.features.Feature`
            Feature to be added.
        """
        self.feature_collection.add_feature_func(feature.feature_func)
        self._add_tf_mapper(feature)

    def add_seeds(self, nsampler):
        if TF_FLAG:

            from acoupipe.writer import int_list_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "seeds" : int_list_feature,})
            if nsampler > 0:
                self.feature_collection.feature_tf_shape_mapper.update({
                                                "seeds" : (nsampler,2)})
            else:
                self.feature_collection.feature_tf_shape_mapper.update({
                                                "seeds" : (None,)})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"seeds" : "uint64"})

    def add_idx(self):
        if TF_FLAG:
            from acoupipe.writer import int64_feature
            self.feature_collection.feature_tf_encoder_mapper.update({
                                            "idx" : int64_feature,})
            self.feature_collection.feature_tf_shape_mapper.update({
                                            "idx" : (),})
            self.feature_collection.feature_tf_dtype_mapper.update(
                                            {"idx" : "uint64"})


