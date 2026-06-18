"""
High-level Monte-Carlo datasets with the Dataset API
====================================================

This tutorial shows how to build a tiny Monte-Carlo simulation with the
high-level :class:`acoupipe.datasets.Dataset` API.

A Monte-Carlo simulation repeats the same model many times while changing one
or more uncertain input parameters. Here the uncertain parameter is the RMS
pressure of a white-noise point source. For every dataset sample, AcouPipe will

1. draw a new RMS value from a Rayleigh distribution,
2. store that value in a parameter object,
3. copy the value into an Acoular ``WNoiseGenerator``, and
4. calculate a beamforming sourcemap for the sampled source strength.

The central idea is that the Acoular model can stay ordinary Acoular code. The
parameter object only needs to contain values that the Dataset API changes or
passes to callbacks: sampled inputs such as ``rms`` and feature settings such as
the center frequency and bandwidth.
"""

from pathlib import Path

import acoular as ac
import numpy as np

from acoupipe.config import TF_FLAG
from acoupipe.datasets import Config, Dataset, ParameterSet

# Acoular can cache intermediate HDF5 files. That is useful for large studies,
# but examples are easier to understand when every run starts from a clean state.
ac.config.h5library = 'h5py'
ac.config.global_caching = 'none'

###############################################################################
# Build the deterministic Acoular model
# -------------------------------------
#
# The Acoular objects describe the acoustic calculation. They are deterministic:
# if ``whitenoise.rms`` stays the same, the sourcemap stays the same. This setup
# is plain Acoular code; it does not need to know anything about datasets yet.
#
# The model is intentionally small:
#
# * one white-noise generator,
# * one point source using this signal,
# * one microphone geometry from Acoular's example files,
# * one rectangular scan grid, and
# * one conventional beamformer.

whitenoise = ac.WNoiseGenerator(sample_freq=51200, seed=10, rms=1.0, num_samples=51200)
microphones = ac.MicGeom(file=str(Path(ac.__file__).parent / 'xml' / 'array_64.xml'))
source = ac.PointSource(signal=whitenoise, mics=microphones)
spectra = ac.PowerSpectra(source=source, block_size=128, window='Hanning')
grid = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.01)
steering = ac.SteeringVector(grid=grid, mics=microphones)
beamformer = ac.BeamformerBase(freq_data=spectra, steer=steering)

###############################################################################
# Describe the Dataset parameters
# -------------------------------
#
# A high-level dataset separates *what the simulation state is* from *how the
# acoustic model is evaluated*. ``ParameterSet`` exposes every entry as a normal
# Python attribute and accepts a simple mapping from parameter names to values.
#
# The mapping only contains values used by the Dataset API below:
#
# * ``rms`` is sampled before every dataset item,
# * ``center_frequency`` and ``bandwidth`` are passed to calculate the source mapping.

parameters = ParameterSet.from_dict(
    {
        'rms': 1.0,
        'center_frequency': 1000.0,
        'bandwidth': 1,
    },
)

###############################################################################
# Create the Dataset configuration
# --------------------------------
#
# ``Config`` owns the parameter object and knows which samplers, prepare
# callbacks, and feature callbacks should run. ``Dataset`` owns the execution
# machinery.

config = Config(parameters=parameters)
dataset = Dataset(config=config)

###############################################################################
# Register a sampling function
# ----------------------------
#
# ``config.sample`` connects a parameter name to a sampling function. During
# dataset generation, AcouPipe calls the function with a NumPy random generator
# and writes the returned value into ``parameters.rms`` before prepare and
# feature callbacks are evaluated.
#
# The ``random_state`` makes the tutorial reproducible. Remove or change it when
# you want independent random sequences.

config.sample(
    'rms',
    random_func=lambda rng: rng.rayleigh(scale=5.0),
    random_state=np.random.default_rng(1),
)

###############################################################################
# Prepare the runtime model
# -------------------------
#
# Samplers modify the parameter object. Prepare callbacks translate the current
# parameter state into the runtime objects used by the simulation. This keeps the
# Monte-Carlo state explicit and avoids reading values back from low-level
# sampler internals.


def apply_parameters_to_runtime(parameters):
    """Copy the sampled RMS value into the mutable Acoular signal object."""
    whitenoise.rms = parameters.rms


config.prepare(apply_parameters_to_runtime)

###############################################################################
# Register dataset features
# -------------------------
#
# A feature callback returns the scalar or array that should appear in the
# generated sample under the registered feature name. Simple callbacks can ignore
# the Dataset internals entirely. If needed, callbacks may declare ``parameters``,
# ``sampler``, or ``data`` arguments and AcouPipe will pass that context.
#
# The ``dtype`` and ``shape`` metadata make the same feature callbacks usable for
# plain Python iteration, HDF5 writing, TensorFlow datasets, and TFRecord files.


def calculate_sourcemap(parameters):
    """Calculate the beamforming result for the current parameter state."""
    sourcemap = beamformer.synthetic(parameters.center_frequency, parameters.bandwidth)
    return sourcemap.astype(np.float32)


config.feature('rms', lambda parameters: np.float32(parameters.rms), dtype=np.float32, shape=())
config.feature('sourcemap', calculate_sourcemap, dtype=np.float32, shape=grid.shape)

###############################################################################
# Generate one sample
# -------------------
#
# ``dataset.generate`` returns an iterator. Each item is a dictionary with the
# registered feature names. Real studies usually set ``size`` to hundreds or
# thousands and write the generated stream to disk. A single sample is enough to
# see the complete control flow.


def run_example():
    """Generate one high-level Monte-Carlo dataset sample."""
    sample = next(dataset.generate(size=1, start_idx=1, progress_bar=False))
    return {
        'rms': sample['rms'],
        'sourcemap': sample['sourcemap'],
    }


result = run_example()
print(f'Sampled RMS value: {result["rms"]:.3f}')  # noqa: T201
print(f'Sourcemap shape: {result["sourcemap"].shape}')  # noqa: T201

###############################################################################
# Create a TensorFlow dataset
# ---------------------------
#
# If TensorFlow is installed, the same configuration can directly create a
# ``tf.data.Dataset``. This is useful when the Acoular model should feed a
# training pipeline without writing an intermediate file.


def get_tensorflow_dataset(size=1):
    """Return a TensorFlow dataset backed by this AcouPipe Dataset."""
    if not TF_FLAG:
        msg = 'TensorFlow is not installed.'
        raise RuntimeError(msg)
    return dataset.get_tf_dataset(
        features=['rms', 'sourcemap'],
        size=size,
        start_idx=1,
        progress_bar=False,
    )


if TF_FLAG:
    tf_sample = next(iter(get_tensorflow_dataset()))
    print(f'TensorFlow sourcemap shape: {tuple(tf_sample["sourcemap"].shape)}')  # noqa: T201

###############################################################################
# Save ML-ready files
# -------------------
#
# The high-level Dataset object can also write the registered features to HDF5
# or TFRecord files. HDF5 is convenient for inspection and exchange. TFRecord is
# useful when TensorFlow training jobs should stream precomputed samples.


def save_example_files(output_dir):
    """Save one sample to HDF5 and, when available, TFRecord files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'h5': output_dir / 'high_level_monte_carlo.h5'}
    dataset.save_h5(
        features=['rms', 'sourcemap'],
        size=1,
        name=paths['h5'],
        start_idx=1,
        progress_bar=False,
    )
    if TF_FLAG:
        paths['tfrecord'] = output_dir / 'high_level_monte_carlo.tfrecord'
        dataset.save_tfrecord(
            features=['rms', 'sourcemap'],
            size=1,
            name=paths['tfrecord'],
            start_idx=1,
            progress_bar=False,
        )
    return paths


###############################################################################
# Where to go next
# ----------------
#
# To adapt this tutorial:
#
# * build your Acoular model as usual,
# * add entries to the ``ParameterSet`` for sampled values and feature settings,
# * register sampling functions with ``config.sample(...)`` for the parameters
#   that should vary from sample to sample,
# * use ``config.prepare(...)`` to apply sampled values to your Acoular model,
# * use ``config.feature(...)`` for every output you want to store or inspect,
#   and
# * increase ``size`` to generate many independent dataset samples.
#
# The important idea is that the parameter object only holds Dataset-level
# state, samplers dynamically mutate selected parameters, prepare callbacks
# update the runtime model, and feature callbacks return data for the current
# sample.  For reusable Feature objects with their own per-Sample prepare step,
# see ``examples/high_level_dataset_features.py``.
