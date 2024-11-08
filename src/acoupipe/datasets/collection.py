

from functools import partial

from acoupipe.config import TF_FLAG


class BaseFeatureCollection:
    """
    BaseFeatureCollection base class for handling feature funcs.

    Attributes
    ----------
    feature_funcs : list
        List of feature_funcs.
    """

    def __init__(self, feature_funcs=None, feature_tf_encoder_mapper=None, feature_tf_shape_mapper=None, feature_tf_dtype_mapper=None):
        if feature_funcs is None:
            self.feature_funcs = []
        if feature_tf_encoder_mapper is None:
            self.feature_tf_encoder_mapper = {}
        if feature_tf_shape_mapper is None:
            self.feature_tf_shape_mapper = {}
        if feature_tf_dtype_mapper is None:
            self.feature_tf_dtype_mapper = {}
        self._signature = {}

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
            feature_dict = {}
            for ffunc in feature_funcs:
                result = ffunc(sampler=sampler)
                feature_dict.update(result)
            #     del result  # Delete result explicitly to free up memory
            # gc.collect()  # Manually trigger garbage collection
            return feature_dict
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
        for feature in features:
            if isinstance(feature, str):
                feature_name = feature
            else:
                feature_name = feature.name
            self._signature[feature_name] = tf.TensorSpec(
                self.feature_tf_shape_mapper[feature_name],self.feature_tf_dtype_mapper[feature_name])
        return self._signature
    BaseFeatureCollection.get_output_signature = get_output_signature


class FeatureCollectionBuilder:
    """
    FeatureCollectionBuilder base class for building a BaseFeatureCollection.

    Attributes
    ----------
    feature_collection : BaseFeatureCollection
        BaseFeatureCollection object.
    """

    def __init__(self, feature_collection=None):
        if feature_collection is None:
            self.feature_collection = BaseFeatureCollection()

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


