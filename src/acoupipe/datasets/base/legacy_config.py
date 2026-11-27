"""Legacy dataset configuration base class."""

from functools import partial

from acoupipe.datasets.features import Feature, combine_feature_funcs, remove_unrequested

from traits.api import HasPrivateTraits


class ConfigBase(HasPrivateTraits):
    """Configuration base class for generating microphone array datasets."""

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.base.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The samplers are invoked in dict order. The key
        identifies the random stream of a sampler, so that its samples do not change if other
        samplers are added or removed.

        Examples
        --------
        >>> ConfigBase().get_sampler()
        {}

        e.g.:

        .. code-block:: python

            sampler = {
                0 : BaseSampler(...),
                1 : BaseSampler(...),
                ...
            }

        Returns
        -------
        dict
            dictionary containing the sampler objects
        """
        return {}

    def _get_default_feature_kwargs(self, f, num):
        """Return keyword arguments passed to the `_get_default_feature_{name}` methods."""
        return {'f': f, 'num': num}

    def get_default_features(self, features, f, num):
        """
        Build default features using `_get_default_feature_{name}` methods.

        Parameters
        ----------
        features : list[str]
            Names of default features to include.
        f : float | list[float] | None
            Frequencies used for frequency-dependent features.
        num : int
            Bandwidth selector for fractional octave features.

        Returns
        -------
        list of Feature
            Feature objects of the requested default features.
        """
        feature_kwargs = self._get_default_feature_kwargs(f, num)
        default_features = []
        for feature_name in features:
            if feature_name != 'idx':
                get_feature = getattr(self, f'_get_default_feature_{feature_name}', None)
                if get_feature is None:
                    msg = f'Unknown feature "{feature_name}".'
                    raise ValueError(msg)
                default_features.append(get_feature(**feature_kwargs))
        return default_features

    def get_prepare_func(self):
        """
        Return the prepare function of the configuration.

        The prepare function is called once per sample, before all feature functions.
        Its outputs (e.g. source locations or source strengths) are passed to all
        feature functions and can be used for their calculations. Afterwards, these
        outputs are removed from the sample unless they are requested as features.

        Subclasses that need a preparation step override this method and return a
        single callable. Several preparation steps should be combined within this
        callable.

        Returns
        -------
        callable or None
            Function with signature ``f(sampler) -> dict``, or None if no
            preparation is needed (default).
        """
        return

    def get_features(self, features, f, num):
        """
        Get the combined feature function and the feature objects of this config.

        The feature functions are called in the following order: the prepare
        function of the config (if any), the default features requested by name,
        the custom feature objects (Feature instances) and finally a cleanup that
        removes all outputs that were not requested (see
        :func:`~acoupipe.datasets.features.remove_unrequested`).

        Parameters
        ----------
        features : list
            List of features included in the dataset. Contains names of default
            features (str) and/or custom feature objects (Feature).
        f : float
            The center frequency or list of frequencies of the dataset.
            If None, all frequencies are included.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

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
        feature_func : callable
            Single combined callable with signature ``f(sampler) -> dict`` (see
            :func:`~acoupipe.datasets.features.combine_feature_funcs`). It chains the
            prepare function, the feature functions of all requested features and the
            cleanup, and returns their merged outputs as one dictionary.
        feature_objects : list of Feature
            Feature objects in the order in which they are calculated.
        """
        feature_funcs = []
        prepare_func = self.get_prepare_func()
        if prepare_func is not None:
            feature_funcs.append(prepare_func)
        default_feature_names = [feat for feat in features if isinstance(feat, str)]
        custom_features = [feat for feat in features if isinstance(feat, Feature)]
        feature_objects = self.get_default_features(default_feature_names, f, num) + custom_features
        feature_funcs += [feature.get_feature_func() for feature in feature_objects]
        keep = default_feature_names + [feature.name for feature in custom_features]
        feature_funcs.append(partial(remove_unrequested, keep=keep))
        return combine_feature_funcs(feature_funcs), feature_objects


__all__ = ['ConfigBase']
