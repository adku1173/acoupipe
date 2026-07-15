"""
Loading Data into TensorFlow / Keras
====================================

Once a dataset has been written to disk it can be streamed into a TensorFlow input
pipeline for training. This example loads a stored HDF5 dataset and turns it into a
``tf.data.Dataset`` that a Keras model can consume, and then shows the analogous path
for a dataset stored in the TFRecord format.
"""

# %%
# First, the necessary Python modules are imported.

from acoupipe.datasets.synthetic import DatasetSynthetic
from acoupipe.loader import LoadH5Dataset

import tensorflow as tf

# %%
# A small dataset is written to an HDF5 file, here using the cross-spectral matrix
# (``csm``) as the input feature. Writing and loading datasets is covered in detail
# in the
# :ref:`HDF5 example <sphx_glr_auto_examples_pipelines_example_pipeline_HDF5.py>`.
# The features ``idx`` and ``seeds`` are always stored alongside the selected ones.

DATASET_NAME = 'tensorflow_dataset.h5'

dataset = DatasetSynthetic()
dataset.save_h5(features=['csm'], split='training', size=5, name=DATASET_NAME, progress_bar=False)

# %%
# The stored dataset is loaded again and exposes a generator over its samples through
# ``get_dataset_generator``.

loader = LoadH5Dataset(name=DATASET_NAME)
data_generator = loader.get_dataset_generator()

# %%
# To build a ``tf.data.Dataset`` from the generator, TensorFlow needs to know the
# structure of the yielded samples, the so called output signature. If the shapes and
# dtypes are known, it can be written out explicitly.

output_signature = {
    'csm': tf.TensorSpec(shape=(None, 64, 64), dtype=tf.complex64),
    'idx': tf.TensorSpec(shape=(), dtype=tf.int64),
    'seeds': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
}

tf_dataset = tf.data.Dataset.from_generator(
    generator=data_generator,
    output_signature=output_signature,
).repeat()

# %%
# Writing the signature by hand requires knowing every shape and dtype in advance.
# More conveniently, the dataset can provide a matching signature automatically via
# ``get_output_signature``.

output_signature = dataset.get_output_signature(features=['csm', 'idx', 'seeds'])

tf_dataset = tf.data.Dataset.from_generator(
    generator=data_generator,
    output_signature=output_signature,
).repeat()

# %%
# The resulting dataset yields dictionaries of tensors and can be iterated, batched
# and fed to a model like any other ``tf.data.Dataset``.

sample = next(iter(tf_dataset))
print(f'sample index: {int(sample["idx"])}')
for key, value in sample.items():
    print(f'{key}: shape {value.shape}, dtype {value.dtype}')

# %%
# The same pipeline can be built from a dataset stored in the TFRecord format. There,
# the parsing is handled by a parser function, which the dataset provides through
# ``get_tfrecord_parser``. See the
# :ref:`TFRecord example <sphx_glr_auto_examples_pipelines_example_pipeline_tfrecord.py>`
# for how such a parser works internally.

TFRECORD_NAME = 'example_dataset.tfrecord'

dataset.save_tfrecord(features=['csm'], split='training', size=5, name=TFRECORD_NAME, progress_bar=False)
parser = dataset.get_tfrecord_parser(features=['csm', 'idx', 'seeds'], f=None, num=0)

tf_dataset = tf.data.TFRecordDataset(filenames=[TFRECORD_NAME]).map(parser)

sample = next(iter(tf_dataset))
print(f'sample index: {int(sample["idx"])}')
for key, value in sample.items():
    print(f'{key}: shape {value.shape}, dtype {value.dtype}')

# %%
# From here either dataset can be batched and passed directly to the ``fit`` method of
# a Keras model, for example ``model.fit(tf_dataset.batch(8), ...)``.