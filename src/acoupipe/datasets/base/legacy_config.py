"""Legacy dataset configuration base class."""

from traits.api import HasPrivateTraits


class ConfigBase(HasPrivateTraits):
    """Configuration base class for generating microphone array datasets."""

    def get_sampler(self):
        """Return dictionary containing the sampler objects of type :class:`acoupipe.base.BaseSampler`.

        this function has to be manually defined in a dataset subclass.
        It includes the sampler objects as values. The key defines the idx in the sample order.

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
            if feature_name not in ['idx', 'seeds']:
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


__all__ = ['ConfigBase']
