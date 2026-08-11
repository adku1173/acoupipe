"""
Writing and Loading a TFRecord Dataset
======================================

This example demonstrates how AcouPipe stores a dataset in the TFRecord file format and, more
importantly, how the stored data is parsed back into usable arrays. A small dataset is created
with the built-in :class:`~acoupipe.datasets.synthetic.DatasetSynthetic` and written to
a ``.tfrecord`` file; the remainder of the example focuses on decoding it again.
"""

# %%
# First, the necessary Python modules are imported. TensorFlow_ is required to read
# TFRecord files.

import acoular as ac
from acoupipe.datasets.synthetic import DatasetSynthetic

import matplotlib.pyplot as plt
import tensorflow as tf

# %%
# A small dataset of five source cases is created with the default
# :class:`~acoupipe.datasets.synthetic.DatasetSynthetic` and written to a ``.tfrecord`` file.
# The conventional  beamforming map is included by selecting the ``sourcemap`` feature, together
# with the source locations and the frequency. Internally this uses
# AcouPipe's :class:`WriteTFRecord <acoupipe.writer>` writer , which works analogously to
# the :class:`WriteH5Dataset <acoupipe.writer.WriteH5Dataset>` writer shown in the
# :ref:`HDF5 example <sphx_glr_auto_examples_pipelines_example_1_pipeline_HDF5.py>`.

DATASET_NAME = 'example_dataset.tfrecord'

dataset = DatasetSynthetic()
dataset.save_tfrecord(
    features=['sourcemap', 'loc', 'f'], f=2000, split='training', size=5, name=DATASET_NAME, progress_bar=False
)

# %%
# To read the data back, a parser function describes how each stored feature is decoded. Variable
# length features such as the flattened sourcemap are read with a `tf.io.VarLenFeature` and
# reshaped back to the grid shape.

grid_shape = dataset.config.grid.shape


def parse_record(record):
    """Decode a single serialized TFRecord example back into features."""
    parsed = tf.io.parse_single_example(
        record,
        {
            'sourcemap': tf.io.VarLenFeature(tf.float32),
            'loc': tf.io.VarLenFeature(tf.float32),
            'f': tf.io.VarLenFeature(tf.float32),
            # and further features ...
        },
    )
    sourcemap = tf.reshape(tf.sparse.to_dense(parsed['sourcemap']), grid_shape)
    loc = tf.reshape(tf.sparse.to_dense(parsed['loc']), (3, -1))
    f = tf.sparse.to_dense(parsed['f'])
    # and further features ...
    return {'sourcemap': sourcemap, 'loc': loc, 'f': f}


# %%
# The file is opened as a ``tf.data.TFRecordDataset`` and the parser is applied to each record. The
# first decoded sample is then retrieved.

tf_dataset = tf.data.TFRecordDataset(filenames=[DATASET_NAME]).map(parse_record)
sample = next(iter(tf_dataset))

# %%
# Finally, the decoded beamforming map of the first sample is displayed together with the true
# source locations.

Lm = ac.L_p(sample['sourcemap'].numpy()).T
extent = dataset.config.grid.extent
f_hz = int(sample['f'].numpy()[0])

fig, ax = plt.subplots()
im = ax.imshow(Lm, origin='lower', vmin=Lm.max() - 15, extent=extent)
for loc in sample['loc'].numpy().T:
    ax.plot(loc[0], loc[1], 'x', color='red')
ax.set_title(f'beamforming map (f = {f_hz} Hz)')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
fig.colorbar(im, ax=ax, label='SPL / dB')
plt.show()

# %%
# Writing the parser by hand shows what happens under the hood. For convenience, the dataset also
# provides a ready made parser via ``dataset.get_tfrecord_parser(...)`` that returns an equivalent
# parser function.
