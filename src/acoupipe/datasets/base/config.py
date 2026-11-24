"""High-level Dataset configuration.

This module provides the high-level :class:`Config` class for ParameterSet-driven
Dataset generation. It enables users to configure dataset generation with
sampleable parameters, custom prepare and feature callbacks, and flexible
pipeline configuration.

.. autosummary::
    :toctree: generated/

    Config
"""

from functools import partial

from acoupipe.base import BaseSampler
from acoupipe.datasets.base.callbacks import call_with_supported_context
from acoupipe.datasets.base.parameters import ParameterSet
from acoupipe.datasets.features import Feature, combine_feature_funcs, create_feature, remove_unrequested
from acoupipe.pipeline import seed_key
from acoupipe.sampler import AttributeSampler

from traits.api import Any, Callable, Dict, HasPrivateTraits, Instance, List, Str


class Config(HasPrivateTraits):
    """High-level configuration for ParameterSet-driven Dataset generation.

    The :class:`Config` class provides a user-friendly interface for configuring
    dataset generation. It manages sampleable parameters, registered callbacks for
    prepare and feature extraction, and pipeline configuration.

    Users can create custom Config subclasses for specific dataset types by
    overriding :meth:`create_parameters` to return dataset-specific parameter sets.

    Parameters
    ----------
    parameters : ParameterSet or dict, optional
        The mutable parameter set for this configuration. If a dict is provided,
        it will be converted to a ParameterSet. If None, a default ParameterSet
        is created by calling :meth:`create_parameters`.
    **traits
        Additional Traits-specific attributes passed to the parent class.

    Attributes
    ----------
    parameters : ParameterSet
        The mutable parameter object containing all configuration parameters
        that can be sampled during dataset generation.

    See Also
    --------
    acoupipe.datasets.base.parameters.ParameterSet : Base class for parameter sets.
    acoupipe.datasets.base.DatasetBase : Base class for datasets.

    Notes
    -----
    The Config class is designed to work alongside the legacy ConfigBase class.
    Existing code using ConfigBase continues to work, while new code can use
    the more flexible Config API.

    During :meth:`acoupipe.datasets.base.DatasetBase.generate`, each generated
    sample follows the same high-level order:

    1. The pipeline re-seeds every sampler from the dataset seed, the split, the sample
       index and the sampler key (see :func:`~acoupipe.pipeline.reseed_sampler`).
    2. Registered parameter samplers are executed in registration order. The key of a
       sampler created with :meth:`sample` is derived from its parameter path, so that its
       samples do not change if other samplers are added or removed.
    3. Dataset-specific :meth:`_global_prepare` and the prepare callbacks registered with
       :meth:`prepare` are evaluated in registration order and may add
       intermediate values to the per-sample ``data`` dictionary.
    4. Requested ParameterSet attributes are copied to ``data``.
    5. Requested default features, feature objects and user features registered
       with :meth:`feature` are evaluated, in this order. They may use both
       sampled parameters and values already present in ``data``.
    6. All outputs that were not requested, e.g. intermediate values of prepare
       callbacks, are removed from the generated sample.

    Users normally configure these steps on the Config object and then let the
    Dataset execute them. Calling a sampler's ``sample()`` method directly is
    useful for small demonstrations, but is not the normal dataset-generation
    workflow.

    Examples
    --------
    Create a configuration with an explicit parameter set:

    >>> from acoupipe.datasets.base.config import Config
    >>> from acoupipe.datasets.base.parameters import ParameterSet
    >>> parameters = ParameterSet(c=343.0, fs=13720.0)
    >>> config = Config(parameters=parameters)
    >>> config.parameters.c
    343.0

    Dataset-specific configuration classes usually provide their own defaults by
    overriding :meth:`create_parameters`.
    """

    #: The mutable parameter set for this configuration.
    parameters = Any(desc='Monte-Carlo parameters')

    #: List of samplers created for parameter paths.
    _parameter_samplers = List(Instance(BaseSampler), desc='samplers created for parameter paths')

    #: List of user-registered prepare callbacks.
    _registered_prepare_funcs = List(Callable, desc='user-registered prepare callbacks')

    #: User-registered features, mapped from feature name to Feature object.
    _registered_features = Dict(Str, Instance(Feature), desc='user-registered features')

    def __init__(self, parameters=None, **traits):
        """Initialize the Config object.

        Parameters
        ----------
        parameters : ParameterSet or dict, optional
            The parameter set for this configuration. If a dict is provided,
            it will be converted to a ParameterSet. If None, a default
            ParameterSet is created by calling :meth:`create_parameters`.
        **traits
            Additional Traits-specific attributes.
        """
        if parameters is None:
            parameters = self.create_parameters()
        super().__init__(parameters=parameters, **traits)
        self.prepare(self._global_prepare)

    def _global_prepare(self):
        """Run dataset-specific preparation before all user prepare callbacks.

        Subclasses override this method and may declare ``parameters``, ``sampler``
        or ``data`` by name, like any :meth:`prepare` callback.
        """

    def create_parameters(self):
        """Create the default parameter set for this configuration.

        Returns
        -------
        ParameterSet
            A new ParameterSet instance with default parameters.

        Notes
        -----
        Subclasses should override this method to return dataset-specific
        parameter sets with appropriate defaults.
        """
        return ParameterSet()

    def _resolve_parameter_path(self, path):
        """Resolve and validate a parameter path.

        Parameters
        ----------
        path : str
            The parameter path to resolve.

        Returns
        -------
        object
            The value at the specified parameter path.

        Raises
        ------
        ValueError
            If the path is invalid or does not exist in the parameters.
        """
        if not isinstance(path, str) or not path or '.' in path or not hasattr(self.parameters, path):
            msg = f'Unsupported sampling path "{path}". The parameter object has no such attribute.'
            raise ValueError(msg)
        return getattr(self.parameters, path)

    def sample(self, path, random_var=None, random_func=None, sampler_class=None, equal_value=True, **sampler_kwargs):
        """Register a sampler for a parameter attribute.

        This method registers an :class:`~acoupipe.sampler.AttributeSampler` that will
        sample the specified parameter during dataset generation.

        The sampler can use either a ``scipy.stats`` random variable or a custom
        random function. A custom function may declare either ``random_func(rng)``
        or ``random_func(rng, parameters)``. The second form injects the active
        parameter set into the random function, which is useful for coupled
        sampling rules. Sample ``data`` is not available at this stage because
        sampling happens before prepare and feature callbacks; use
        :meth:`prepare` when a callback needs the sample data context.

        Parameters
        ----------
        path : str
            The name of the parameter attribute to sample.
        random_var : scipy.stats distribution, optional
            A scipy.stats random variable for sampling. Either this or
            `random_func` must be provided.
        random_func : callable, optional
            A custom random function for sampling. Either this or `random_var`
            must be provided. Supported signatures are ``random_func(rng)`` and
            ``random_func(rng, parameters)``. In the second form, `parameters`
            is the active :class:`~acoupipe.datasets.base.parameters.ParameterSet`.
        sampler_class : type, optional
            The sampler class to use. Currently only AttributeSampler is supported.
        equal_value : bool, optional
            If True, the same value is used for all samples. Default is True.
        **sampler_kwargs
            Additional keyword arguments passed to the sampler constructor.

        Returns
        -------
        AttributeSampler
            The registered sampler instance.

        Raises
        ------
        ValueError
            If the path is invalid, or if neither random_var nor random_func
            is provided, or if an unsupported sampler_class is specified.

        Examples
        --------
        Imports only need to appear once inside a docstring example block; all
        examples in this block share the same doctest namespace.

        >>> import numpy as np
        >>> from scipy.stats import norm
        >>> from acoupipe.datasets.base import DatasetBase
        >>> from acoupipe.datasets.base.parameters import ParameterSet

        Register a sampler using a ``scipy.stats`` distribution:

        >>> parameters = ParameterSet(c=343.0, fs=13720.0)
        >>> config = Config(parameters=parameters)
        >>> c_sampler = config.sample('c', random_var=norm(loc=340.0, scale=0.0))
        >>> c_sampler.random_state = np.random.default_rng(1)
        >>> float(c_sampler.sample())
        340.0
        >>> float(parameters.c)
        340.0

        Register a sampler using a NumPy random function. The callable receives
        the sampler's random number generator as first argument:

        >>> parameters = ParameterSet(c=343.0)
        >>> config = Config(parameters=parameters)
        >>> def random_c(rng):
        ...     return rng.normal(loc=343.0, scale=2.0)
        >>> c_sampler = config.sample('c', random_func=random_c)
        >>> c_sampler.random_state = np.random.default_rng(1)
        >>> round(float(c_sampler.sample()), 3)
        343.691

        Register two independent samplers when two parameters should be sampled
        for every generated sample. Calling :meth:`sample` only registers the
        sampling rule; users normally do not call the returned sampler's
        ``sample()`` method themselves. During dataset generation,
        :class:`~acoupipe.datasets.base.DatasetBase` configures the underlying
        pipeline. For each generated sample, the pipeline assigns random states,
        calls the registered samplers, and then evaluates prepare and feature
        callbacks.

        >>> parameters = ParameterSet(c=343.0, fs=13720.0)
        >>> config = Config(parameters=parameters)
        >>> _ = config.sample('c', random_var=norm(loc=340.0, scale=0.0))
        >>> _ = config.sample('fs', random_var=norm(loc=48000.0, scale=0.0))
        >>> dataset = DatasetBase(config=config)
        >>> sample = next(dataset.generate(features=['c', 'fs'], size=1, progress_bar=False))
        >>> float(sample['c']), float(sample['fs'])
        (340.0, 48000.0)

        The pipeline samples registered samplers in registration order, so the
        example above samples ``c`` before ``fs``. Each parameter path can be
        sampled by one sampler only.

        A random function may also request the current parameter set. This allows
        one sampler to draw a coupled pair: it returns the value for its own path
        and updates another parameter at the same time.

        >>> parameters = ParameterSet(c=343.0, fs=13720.0)
        >>> config = Config(parameters=parameters)
        >>> def random_acoustic_pair(rng, parameters):
        ...     c = rng.normal(loc=340.0, scale=0.0)
        ...     parameters.fs = c * 40.0
        ...     return c
        >>> c_sampler = config.sample('c', random_func=random_acoustic_pair)
        >>> c_sampler.random_state = np.random.default_rng(1)
        >>> float(c_sampler.sample())
        340.0
        >>> float(parameters.c), float(parameters.fs)
        (340.0, 13600.0)

        See Also
        --------
        prepare : Register callbacks with full context injection (parameters, sampler, data).
        feature : Register feature callbacks with full context injection.
        """
        self._resolve_parameter_path(path)
        if any(sampler.attribute == path for sampler in self._parameter_samplers):
            msg = f'Sampling path "{path}" is already sampled.'
            raise ValueError(msg)
        if sampler_class not in (None, AttributeSampler):
            msg = f'Sampling path "{path}" supports only AttributeSampler.'
            raise ValueError(msg)
        if random_var is None and random_func is None:
            msg = f'Sampling path "{path}" requires random_var or random_func.'
            raise ValueError(msg)
        sampler = AttributeSampler(
            target=self.parameters,
            attribute=path,
            parameters=self.parameters,
            random_var=random_var,
            random_func=random_func,
            equal_value=equal_value,
            **sampler_kwargs,
        )
        self._parameter_samplers.append(sampler)
        return sampler

    def prepare(self, func):
        """Register a prepare callback that runs after sampling and before feature extraction.

        Prepare callbacks can be used to perform setup operations that depend on
        the sampled parameter values. They receive the parameters, sampler, and
        data as arguments via the callback context injection pattern.

        The callback function can access and modify the data dictionary, and
        return values will be merged into the data context for subsequent
        feature calculations.

        Parameters
        ----------
        func : callable
            The prepare function to register. It should accept keyword arguments
            `parameters`, `sampler`, and `data`, and return a dict that will be
            merged into the data, or None.

        Returns
        -------
        callable
            The registered function (for decorator usage).

        Examples
        --------
        Imports only need to appear once inside a docstring example block; all
        examples in this block share the same doctest namespace.

        >>> from acoupipe.datasets.base import DatasetBase
        >>> from acoupipe.datasets.base.parameters import ParameterSet

        Prepare callbacks are useful for intermediate quantities that should be
        computed after parameter sampling but before feature calculation. They
        can request ``parameters`` and ``data`` by name. Returned dictionaries are
        merged into the per-sample ``data`` dictionary.

        >>> parameters = ParameterSet(c=340.0, fs=13600.0)
        >>> config = Config(parameters=parameters)
        >>> @config.prepare
        ... def add_wavelength(parameters, data):
        ...     return {'wavelength': parameters.c / parameters.fs}
        >>> def wavelength_feature(data):
        ...     return data['wavelength']
        >>> _ = config.feature('wavelength', wavelength_feature)
        >>> dataset = DatasetBase(config=config)
        >>> sample = next(dataset.generate(features=['wavelength'], size=1, progress_bar=False))
        >>> float(sample['wavelength'])
        0.025

        Multiple prepare callbacks are executed in registration order. Later
        callbacks can use values added to ``data`` by earlier callbacks.

        >>> parameters = ParameterSet(c=340.0, fs=13600.0)
        >>> config = Config(parameters=parameters)
        >>> @config.prepare
        ... def add_center_frequency(parameters, data):
        ...     return {'center_frequency': parameters.fs / 4.0}
        >>> @config.prepare
        ... def add_wavelength(parameters, data):
        ...     return {'wavelength': parameters.c / data['center_frequency']}
        >>> def wavelength_feature(data):
        ...     return data['wavelength']
        >>> _ = config.feature('wavelength', wavelength_feature)
        >>> dataset = DatasetBase(config=config)
        >>> sample = next(dataset.generate(features=['wavelength'], size=1, progress_bar=False))
        >>> float(sample['wavelength'])
        0.1
        """

        def prepare_func(sampler, data):
            result = call_with_supported_context(
                func,
                parameters=self.parameters,
                sampler=sampler,
                data=data,
            )
            return result if isinstance(result, dict) else {}

        self._registered_prepare_funcs.append(prepare_func)
        return func

    def feature(self, name, func, dtype=None, shape=None):
        """Register a named feature callback for Dataset generation.

        Feature callbacks are used to compute custom features from the sampled
        parameters and other data. They run after prepare callbacks.

        The callback function receives parameters, sampler, and data via the
        callback context injection pattern, allowing it to access sampled values
        and compute derived features.

        Parameters
        ----------
        name : str
            The name of the feature to register.
        func : callable
            The feature function. It should accept keyword arguments `parameters`,
            `sampler`, and `data`, and return the feature value.
        dtype : numpy.dtype, optional
            The data type of the feature. Required together with shape for
            TensorFlow datasets and TFRecord files.
        shape : tuple, optional
            The shape of the feature. Required together with dtype for
            TensorFlow datasets and TFRecord files.

        Returns
        -------
        callable
            The registered function.

        Raises
        ------
        ValueError
            If a feature with the same name is already registered.

        Examples
        --------
        Imports only need to appear once inside a docstring example block; all
        examples in this block share the same doctest namespace.

        >>> import numpy as np
        >>> from acoupipe.datasets.base.parameters import ParameterSet

        Register a feature to compute a custom value from parameters:

        >>> parameters = ParameterSet(snr_db=20.0)
        >>> config = Config(parameters=parameters)
        >>> def compute_snr_linear(parameters, data, sampler=None):
        ...     return 10 ** (parameters.snr_db / 10)
        >>> config.feature('snr_linear', compute_snr_linear, dtype=np.float64, shape=()) is compute_snr_linear
        True

        Register a feature that depends on data from prepare callbacks:

        >>> parameters = ParameterSet(c=343.0, fs=13720.0)
        >>> config = Config(parameters=parameters)
        >>> @config.prepare
        ... def compute_intermediate(parameters, data):
        ...     return {'intermediate': parameters.c * 2}
        >>> def compute_derived(parameters, data):
        ...     return data.get('intermediate', 0) + parameters.fs
        >>> config.feature('derived', compute_derived, dtype=np.float64, shape=()) is compute_derived
        True
        """
        if name in self._registered_features:
            msg = f'Feature "{name}" is already registered.'
            raise ValueError(msg)

        def feature_func(sampler, data):
            value = call_with_supported_context(
                func,
                parameters=self.parameters,
                sampler=sampler,
                data=data,
            )
            return {name: value}

        self._registered_features[name] = create_feature(feature_func, name=name, shape=shape, dtype=dtype)
        return func

    def get_sampler(self):
        """Return the sampler dictionary for Pipeline execution.

        Returns
        -------
        dict
            A dictionary in registration order that maps the key of each sampler's parameter
            path (see :func:`~acoupipe.pipeline.seed_key`) to the sampler instance.
        """
        return {seed_key(sampler.attribute): sampler for sampler in self._parameter_samplers}

    def _get_parameter_feature(self, name):
        """Return a feature function for a direct ParameterSet attribute.

        This method creates a zero-processing feature that simply copies the
        current value of a parameter into the output.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        callable or None
            A feature function that returns the parameter value, or None if
            the parameter does not exist.
        """
        if not isinstance(name, str) or '.' in name or not hasattr(self.parameters, name):
            return None

        def parameter_feature(sampler):  # noqa: ARG001
            return {name: getattr(self.parameters, name)}

        return parameter_feature

    def get_features(self, features):
        """Get the combined feature function and the feature objects of this config.

        The feature functions are called in the following order:
        prepare callbacks (starting with dataset-specific :meth:`_global_prepare`),
        the requested ParameterSet attributes, the requested Features (default features,
        custom feature objects and registered feature callbacks) and
        finally a cleanup that removes all outputs that were not requested
        (see :func:`~acoupipe.datasets.features.remove_unrequested`). Within each
        of these groups, the functions are called in the order of ``features``.

        Parameters
        ----------
        features : list of str or Feature, optional
            The list of feature names or feature objects to include.

        Returns
        -------
        feature_func : callable
            Single combined callable with signature ``f(sampler) -> dict`` (see
            :func:`~acoupipe.datasets.features.combine_feature_funcs`).
        feature_objects : list of Feature
            Feature objects providing ``name``, ``dtype`` and ``shape`` metadata
            (e.g. for :func:`~acoupipe.datasets.features.get_tf_mappers`). This
            includes registered feature callbacks with ``dtype`` and ``shape``.
        """
        features = [] if features is None else list(features)
        requested = [feat.name if isinstance(feat, Feature) else feat for feat in features]
        parameter_feature_funcs = []
        default_feature_names = []

        # resolve strings for ParameterSet attributes and default Features.
        for feature_name in [feat for feat in features if isinstance(feat, str)]:
            if feature_name == 'idx' or feature_name in self._registered_features:
                continue
            parameter_feature = self._get_parameter_feature(feature_name)
            if parameter_feature is None:
                default_feature_names.append(feature_name)
            else:
                parameter_feature_funcs.append(parameter_feature)

        # default Features followed by Feature objects passed directly by the user
        feature_objects = self.get_default_features(default_feature_names)
        feature_objects += [feat for feat in features if isinstance(feat, Feature)]

        # Call order per sample:
        # 1. prepare callbacks
        feature_funcs = list(self._registered_prepare_funcs)
        # 2. ParameterSet attributes
        feature_funcs += parameter_feature_funcs
        # 3. default Features, then user Feature objects
        feature_funcs += [feature.get_feature_func() for feature in feature_objects]
        # 4. requested registered Features
        for name in requested:
            if name not in self._registered_features:
                continue
            feature = self._registered_features[name]
            feature_funcs.append(feature.get_feature_func())
            # Features without metadata are not passed on to the TF paths yet.
            if feature.dtype is not None and feature.shape is not None:
                feature_objects.append(feature)
        # 5. cleanup
        feature_funcs.append(partial(remove_unrequested, keep=requested))
        return combine_feature_funcs(feature_funcs), feature_objects

    def get_default_features(self, features):
        """Build default features using `_get_default_feature_{name}` methods.

        For each feature name, this method looks for a corresponding
        `_get_default_feature_{name}` method and calls it.

        Parameters
        ----------
        features : list of str
            The list of feature names to build.

        Returns
        -------
        list of Feature
            Feature objects of the requested default features.

        Raises
        ------
        ValueError
            If a feature name does not have a corresponding `_get_default_feature_{name}` method.
        """
        default_features = []
        for feature_name in features:
            if feature_name != 'idx':
                get_feature = getattr(self, f'_get_default_feature_{feature_name}', None)
                if get_feature is None:
                    msg = f'Unknown feature "{feature_name}".'
                    raise ValueError(msg)
                default_features.append(get_feature())
        return default_features


__all__ = ['Config']
