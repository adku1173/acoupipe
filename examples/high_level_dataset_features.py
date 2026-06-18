"""
Adding Features to high-level Monte-Carlo Datasets
==================================================

This user guide is a follow-up to
``examples/high_level_monte_carlo_dataset.py``.  The introductory example shows
how to build a high-level Monte-Carlo :class:`acoupipe.datasets.Dataset`.  This
example keeps the same kind of Acoular simulation, but focuses only on the
available ways to add new **Features** to that Dataset.

In AcouPipe terminology, a **Feature** is the value stored in each generated
**Sample**.  A **Feature object** or callback is the Python logic that computes
that value.  The four routes below are shown in ascending order of API knowledge
needed by the user:

1. register a callback with :meth:`acoupipe.datasets.Config.feature`,
2. wrap a feature function with :func:`acoupipe.datasets.features.create_feature`,
3. define a reusable :class:`acoupipe.datasets.features.BaseFeatureCatalog`
   subclass, and
4. subclass :class:`acoupipe.datasets.Config` and add a
   ``_get_default_feature_<name>`` method so the Feature can be requested by
   string name.

The final section discusses the implications for plain iteration, HDF5,
TensorFlow datasets, and TFRecord files.
"""

from functools import partial
from pathlib import Path

import acoular as ac
import numpy as np
from traits.api import Any, Instance, Tuple

from acoupipe.config import TF_FLAG
from acoupipe.datasets import Config, Dataset, ParameterSet
from acoupipe.datasets.features import BaseFeatureCatalog, create_feature

# Keep examples reproducible and independent from Acoular's persistent cache.
ac.config.h5library = 'h5py'
ac.config.global_caching = 'none'

###############################################################################
# Start from the same Monte-Carlo ingredients
# -------------------------------------------
#
# The introductory example already explains the Dataset control flow.  Here we
# only need a compact version of the same setup: an Acoular signal source whose
# RMS is sampled per Sample, a beamformer, and a ``ParameterSet`` that exposes
# Dataset-level state to callbacks.

whitenoise = ac.WNoiseGenerator(sample_freq=51200, seed=10, rms=1.0, num_samples=51200)
microphones = ac.MicGeom(file=str(Path(ac.__file__).parent / 'xml' / 'array_64.xml'))
source = ac.PointSource(signal=whitenoise, mics=microphones)
spectra = ac.PowerSpectra(source=source, block_size=128, window='Hanning')
grid = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.02)
steering = ac.SteeringVector(grid=grid, mics=microphones)
beamformer = ac.BeamformerBase(freq_data=spectra, steer=steering)

parameters = ParameterSet.from_dict(
    {
        'rms': np.float32(1.0),
        'reference_rms': np.float32(1.0),
        'center_frequency': 1000.0,
        'bandwidth': 1,
        'sourcemap': ParameterSet(c=343.0),
    },
)


def apply_parameters_to_runtime(parameters):
    """Copy sampled Dataset state into the mutable Acoular runtime object."""
    whitenoise.rms = parameters.rms


###############################################################################
# Way 1: register a high-level callback
# -------------------------------------
#
# ``config.feature(...)`` is the easiest route and should be the first choice for
# one-off Features in new high-level Dataset code.  The callback returns the
# Feature value directly.  It may declare only the context it needs; here it asks
# for ``parameters`` and returns the sampled RMS value for the current Sample.
#
# Passing ``dtype`` and ``shape`` is optional for plain Python iteration and HDF5
# writing, but it is required when the Feature should be available through
# TensorFlow datasets, TFRecord writing, or TFRecord parsing.


def add_rms_feature(config):
    """Register the sampled RMS as a high-level Feature callback."""
    config.feature(
        'rms',
        lambda parameters: np.float32(parameters.rms),
        dtype=np.float32,
        shape=(),
    )


###############################################################################
# Way 2: pass a wrapped feature function directly
# -----------------------------------------------
#
# ``create_feature(...)`` wraps a low-level feature function in a Feature object.
# The function receives the private sampler dictionary used by the Pipeline and
# must return a mapping from Feature name to Feature value.  For high-level code,
# prefer closing over ``parameters`` instead of reading hidden sampler internals.
#
# This route is useful when you want to reuse a function in multiple feature
# lists without creating a class.  It requires more API knowledge than
# ``config.feature(...)`` because the function must return a dictionary and the
# Feature object must be passed in the ``features`` list.


def calculate_rms_db(sampler, parameters):  # noqa: ARG001
    """Return the sampled RMS in decibels as a Feature dictionary."""
    return {'rms_db': np.float32(20.0 * np.log10(parameters.rms))}


rms_db_feature = create_feature(
    feature_func=partial(calculate_rms_db, parameters=parameters),
    name='rms_db',
    dtype=np.float32,
    shape=(),
)

###############################################################################
# Way 3: pass a custom Feature object directly
# --------------------------------------------
#
# A custom ``BaseFeatureCatalog`` subclass is the structured version of a wrapped
# function.  It is useful when the Feature object has its own configuration and
# should be shared across projects.  Subclassing requires Traits-compatible
# attributes and a ``get_feature_func`` method, so it is more advanced than
# ``create_feature(...)``.
#
# This Feature object calculates a sourcemap from the current Monte-Carlo state
# and normalizes it by its maximum value.  It demonstrates the Feature lifecycle:
# ``get_prepare_func`` updates Feature-specific Acoular runtime state from the
# current Dataset parameters immediately before ``get_feature_func`` calculates
# the Feature value.


class NormalizedSourcemapFeature(BaseFeatureCatalog):
    """Feature object that calculates a normalized beamforming sourcemap."""

    name = 'normalized_sourcemap'
    dtype = np.float32
    shape = Tuple()
    beamformer = Instance(ac.BeamformerBase)
    parameters = Any()

    def get_prepare_func(self):
        """Return a per-Sample prepare callback for Feature-specific state."""

        def prepare_normalized_sourcemap(parameters, beamformer):
            beamformer.steer.env.c = parameters.sourcemap.c

        return partial(prepare_normalized_sourcemap, beamformer=self.beamformer)

    def get_feature_func(self):
        """Return the low-level feature function used by the Pipeline."""

        def calculate_normalized_sourcemap(sampler, beamformer, parameters, name):  # noqa: ARG001
            sourcemap = beamformer.synthetic(parameters.center_frequency, parameters.bandwidth)
            scale = np.max(np.abs(sourcemap))
            if scale == 0.0:
                return {name: np.zeros_like(sourcemap, dtype=np.float32)}
            return {name: (sourcemap / scale).astype(np.float32)}

        return partial(
            calculate_normalized_sourcemap,
            beamformer=self.beamformer,
            parameters=self.parameters,
            name=self.name,
        )


normalized_sourcemap_feature = NormalizedSourcemapFeature(
    beamformer=beamformer,
    parameters=parameters,
    shape=grid.shape,
)

###############################################################################
# Way 4: subclass Config for string-addressable default Features
# -------------------------------------------------------------
#
# This route follows the same convention that AcouPipe's built-in configs use:
# when the user requests ``'rms_ratio'``, ``Config.get_default_features`` looks
# for a method named ``_get_default_feature_rms_ratio`` and calls it.
#
# This route needs the most API knowledge, but it is the most integrated option
# for reusable Dataset variants.  The user can request the Feature by string, so
# the same name works in iteration, TensorFlow output signatures, HDF5 writing,
# TFRecord writing, and TFRecord parsing.


class FeatureGuideConfig(Config):
    """Config with a string-addressable default Monte-Carlo Feature."""

    def _get_default_feature_rms_ratio(self, **kwargs):  # noqa: ARG002
        """Return sampled RMS relative to ``parameters.reference_rms``."""

        def calculate_rms_ratio(sampler, parameters):  # noqa: ARG001
            value = parameters.rms / parameters.reference_rms
            return {'rms_ratio': np.float32(value)}

        return create_feature(
            feature_func=partial(calculate_rms_ratio, parameters=self.parameters),
            name='rms_ratio',
            dtype=np.float32,
            shape=(),
        )


###############################################################################
# Assemble the Dataset used below
# -------------------------------
#
# The combined Dataset uses the subclassed Config so all four routes can be
# exercised together.  The sampler and prepare callback are the same concepts as
# in the introductory Monte-Carlo example.

config = FeatureGuideConfig(parameters=parameters)
config.sample(
    'rms',
    random_func=lambda rng: np.float32(rng.rayleigh(scale=5.0)),
    random_state=np.random.default_rng(1),
)
config.prepare(apply_parameters_to_runtime)
add_rms_feature(config)
dataset = Dataset(config=config)

all_features = [
    'rms',
    rms_db_feature,
    normalized_sourcemap_feature,
    'rms_ratio',
]

string_addressable_features = ['rms', 'rms_ratio']


###############################################################################
# Generate a Sample with all four Features
# ----------------------------------------
#
# String-addressable Features (``'rms'`` and ``'rms_ratio'``) and direct Feature
# objects can be mixed for plain Dataset iteration.


def run_example():
    """Generate one Monte-Carlo Sample containing all four tutorial Features."""
    sample = next(dataset.generate(features=all_features, size=1, start_idx=1, progress_bar=False))
    return {
        'rms': sample['rms'],
        'rms_db': sample['rms_db'],
        'normalized_sourcemap': sample['normalized_sourcemap'],
        'rms_ratio': sample['rms_ratio'],
    }


result = run_example()
print(f'sampled RMS: {result["rms"]:.3f}')  # noqa: T201
print(f'RMS in dB: {result["rms_db"]:.3f}')  # noqa: T201
print(f'normalized sourcemap shape: {result["normalized_sourcemap"].shape}')  # noqa: T201
print(f'RMS ratio: {result["rms_ratio"]:.3f}')  # noqa: T201

###############################################################################
# Compatibility with HDF5, TensorFlow, and TFRecord
# -------------------------------------------------
#
# Plain iteration and HDF5 writing are permissive because they consume the
# dictionaries yielded by the Pipeline.  All four routes work with
# ``dataset.generate(...)`` and ``dataset.save_h5(...)`` as long as the Feature
# values can be stored by HDF5.
#
# TensorFlow datasets and TFRecord parsing need static metadata.  The
# high-level ``config.feature(...)`` route and the ``_get_default_feature_<name>``
# route are string-addressable, so they are the safest choices for ML pipelines:
# use ``features=['rms', 'rms_ratio']`` and AcouPipe can build the output
# signature or parser from Feature names.
#
# Direct Feature objects created with ``create_feature(...)`` or a custom
# ``BaseFeatureCatalog`` subclass carry dtype and shape metadata, so TFRecord
# writing can encode them.  However, current ``get_tf_dataset(...)`` and
# ``get_tfrecord_parser(...)`` paths expect string feature names when building
# output signatures and parse schemas.  For export-ready reusable Features,
# prefer registering the Feature on a Config subclass so users can request it by
# name.


def save_h5_with_all_features(output_dir):
    """Write all four Features to an HDF5 file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'high_level_dataset_features.h5'
    dataset.save_h5(features=all_features, size=1, name=path, start_idx=1, progress_bar=False)
    return path


def get_tensorflow_dataset_with_string_features(size=1):
    """Return a TensorFlow dataset for the string-addressable Features."""
    if not TF_FLAG:
        msg = 'TensorFlow is not installed.'
        raise RuntimeError(msg)
    return dataset.get_tf_dataset(
        features=string_addressable_features,
        size=size,
        start_idx=1,
        progress_bar=False,
    )


def save_tfrecord_with_all_features(output_dir):
    """Write all four Features to TFRecord when TensorFlow is available."""
    if not TF_FLAG:
        msg = 'TensorFlow is not installed.'
        raise RuntimeError(msg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'high_level_dataset_features.tfrecord'
    dataset.save_tfrecord(features=all_features, size=1, name=path, start_idx=1, progress_bar=False)
    return path


def try_tensorflow_dataset_with_direct_feature_object():
    """Return the documented compatibility result for direct Feature objects."""
    if not TF_FLAG:
        return {
            'api': 'get_tf_dataset',
            'compatible': None,
            'reason': 'TensorFlow is not installed.',
        }
    try:
        next(
            iter(
                dataset.get_tf_dataset(
                    features=[rms_db_feature],
                    size=1,
                    start_idx=1,
                    progress_bar=False,
                ),
            ),
        )
    except KeyError:
        return {
            'api': 'get_tf_dataset',
            'compatible': False,
            'reason': 'A direct Feature object is not a string Feature name in the TensorFlow output signature path.',
        }
    return {
        'api': 'get_tf_dataset',
        'compatible': True,
        'reason': 'Direct Feature object unexpectedly worked with get_tf_dataset.',
    }


if TF_FLAG:
    tf_sample = next(iter(get_tensorflow_dataset_with_string_features()))
    print(f'TensorFlow RMS ratio shape: {tuple(tf_sample["rms_ratio"].shape)}')  # noqa: T201

###############################################################################
# Which route should you choose?
# ------------------------------
#
# * Use ``config.feature(...)`` for simple, high-level custom Features that read
#   from ``ParameterSet`` or previously computed Feature data.
# * Use ``create_feature(...)`` when you need a small reusable Feature object and
#   plain iteration or file writing is enough.
# * Use a custom ``BaseFeatureCatalog`` subclass when the Feature object has
#   reusable configuration or non-trivial setup, such as references to Acoular
#   beamformers or grids.
# * Use a ``Config`` subclass with ``_get_default_feature_<name>`` when the
#   Feature should become part of a Dataset variant's named Feature vocabulary,
#   especially for TensorFlow datasets and TFRecord parsing.
