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
        """Return keyword arguments passed to default feature builder methods."""
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
        list
            Instantiated feature catalog objects.
        """
        builder_kwargs = self._get_default_feature_kwargs(f, num)
        default_features = []
        for feature_name in features:
            if feature_name not in ['idx', 'seeds']:
                builder = getattr(self, f'_get_default_feature_{feature_name}', None)
                if builder is None:
                    msg = f'Unknown feature "{feature_name}".'
                    raise ValueError(msg)
                default_features.append(builder(**builder_kwargs))
        return default_features


__all__ = ['ConfigBase']
