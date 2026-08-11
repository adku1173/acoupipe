"""
Writing and Loading an HDF5 Dataset
===================================

AcouPipe can save the features produced by a processing pipeline to disk and load them back later.
This example builds a small pipeline, writes its output to an HDF5 (``.h5``) file with
a :class:`~acoupipe.writer.WriteH5Dataset` writer, and then reads it back in with
a :class:`~acoupipe.loader.LoadH5Dataset` loader.
"""

# %%
# First, the necessary Python modules and objects are imported.

from pathlib import Path

import acoular as ac
from acoupipe.loader import LoadH5Dataset
from acoupipe.pipeline import BasePipeline
from acoupipe.sampler import NumericAttributeSampler
from acoupipe.writer import WriteH5Dataset

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# %%
# The HDF5 backend and the name of the dataset file are configured. The file is
# written into the current working directory.

ac.config.h5library = 'h5py'
ac.config.global_caching = 'none'
DATASET_NAME = 'example_dataset.h5'

# %%
# A sampler randomizes the ``rms`` of a white noise signal, drawing values from a
# Rayleigh distribution with a fixed random state for reproducibility.

rng = np.random.RandomState(1)
rayleigh_dist = scipy.stats.rayleigh(scale=5.0)
wn = ac.WNoiseGenerator(sample_freq=51200, seed=10, rms=1.0, num_samples=51200)
rms_sampling = NumericAttributeSampler(random_var=rayleigh_dist, target=[wn], attribute='rms', random_state=rng)

# %%
# A standard Acoular_ beamforming chain maps the point source onto a grid. See
# :ref:`an example setup <sphx_glr_auto_examples_introductory_examples_example_3_point_source.py>`
# for a more detailed explanation.

mg = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_64.xml')
p1 = ac.PointSource(signal=wn, mics=mg, loc=[0.0, 0.0, 0.3])
ps = ac.PowerSpectra(source=p1, block_size=128, window='Hanning')
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# The feature extraction function defines what is stored for each sample: here the
# sampled ``rms`` value and the beamforming map (``sourcemap``) at 1000 Hz.


def extract_features(sampler, beamformer, noise):
    """Return the features that are stored for each generated sample."""
    return {'rms': noise.rms, 'sourcemap': beamformer.synthetic(4000, 1)}


# %%
# The :class:`BasePipeline <acoupipe.datasets.base.BasePipeline>` ties the sampler and the feature
# function together. Here, five samples are generated.

pipeline = BasePipeline(sampler={1: rms_sampling}, numsamples=5, features=(extract_features, bb, wn))

# %%
# The pipeline output is written to the ``.h5`` file. Some additional metadata is attached and
# stored alongside the features.

metadata = {'sample_freq': 51200, 'freq': 4000, 'bandwidth': 'octave', 'block_size': 128}

writer = WriteH5Dataset(
    source=pipeline,
    name=DATASET_NAME,
    features=['sourcemap', 'rms'],
    metadata=metadata,
)
writer.save(progress_bar=False)

# %%
# The dataset can now be loaded back from the file. The loader exposes the number of stored samples
# and the names of the available features.

loader = LoadH5Dataset(name=DATASET_NAME)
print(f'number of samples: {loader.numsamples}')
print(f'features: {loader.features}')

# %%
# Finally, the first stored sample is retrieved and its beamforming map is displayed.

sample = next(loader.get_data())
Lm = ac.L_p(sample['sourcemap'])

fig, ax = plt.subplots()
im = ax.imshow(Lm.T, origin='lower', vmin=Lm.max() - 15, extent=rg.extent, interpolation='bicubic')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
fig.colorbar(im, ax=ax, label='SPL / dB')
plt.show()

# %%
# Several writers can be chained to split features across multiple files. The
# same pipeline output can also be written to TFRecord files for use with TensorFlow
# :ref:`(an example setup) <sphx_glr_auto_examples_pipelines_example_2_pipeline_tfrecord.py>`.
