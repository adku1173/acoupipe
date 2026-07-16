"""
Model Training with Data Generated on the Fly
=============================================

If data generation is fast enough, AcouPipe can feed a model during training without
saving anything to disk. This example trains a small neural network for single source
localization, generating the training data on the fly from
:class:`~acoupipe.datasets.synthetic.DatasetSynthetic`.
"""

# %%
# First, the necessary Python modules are imported.

import acoular as ac
from acoupipe.datasets.synthetic import DatasetSynthetic

import matplotlib.pyplot as plt
import tensorflow as tf

# %%
# A synthetic dataset restricted to a single source is created. The fast ``wishart``
# mode samples the cross spectral matrix directly without simulating time data, the
# grid is coarsened to 32 x 32, and ``tasks=1`` runs the generation in process.

dataset = DatasetSynthetic(max_nsources=1, mode='wishart', tasks=1)
dataset.config.grid.increment = 1 / 31  # 32 x 32 grid

# %%
# The dataset provides a ``tf.data.Dataset`` directly through ``get_tf_dataset``,
# yielding the beamforming map as input and the source location as label. A large size
# makes the training stream effectively endless; the number of samples actually used is
# controlled by the number of training steps below.

training_dataset = dataset.get_tf_dataset(features=['sourcemap', 'loc'], f=1000, split='training', size=1000000)
validation_dataset = dataset.get_tf_dataset(features=['sourcemap', 'loc'], f=1000, split='validation', size=16)

# %%
# A small preprocessing function normalizes each beamforming map, adds a channel axis
# and keeps only the x and y coordinates of the source as the label. The datasets are
# then mapped, batched and prefetched.


def prepare(data):
    """Turn a raw sample into a (normalized map, xy location) training pair."""
    feature = data['sourcemap'][0]
    feature = feature / tf.reduce_max(feature)
    feature = feature[..., tf.newaxis]
    label = data['loc'][:2, 0]
    return feature, label


training_dataset = training_dataset.map(prepare).batch(16).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(prepare).batch(16).cache()

# %%
# A compact convolutional network regresses the two source coordinates from the
# 32 x 32 beamforming map. For real tasks a larger architecture can be used, at the
# cost of much longer training.

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2),
    ]
)
model.compile(optimizer=tf.optimizers.Adam(), loss='mse')

# %%
# The model is now trained on the generated data. Because this example is executed
# live while the documentation is built, the training is deliberately kept short
# to illustrate the principle. A model trained this briefly is nowhere near converged,
# so the prediction shown below is only a rough estimate and mainly serves to demonstrate
# how a trained model is applied. Accurate localization would require many more epochs and steps,
# and typically a larger network.

model.fit(training_dataset, validation_data=validation_dataset, epochs=1, steps_per_epoch=50, verbose=0)

# %%
# Finally, the trained model predicts the source location of a single test sample. As
# noted above, the very short training means the prediction will not sit exactly on the
# true source. It is included here only to show how the trained model is used, not as a
# measure of accuracy.

test_dataset = dataset.get_tf_dataset(features=['sourcemap', 'loc'], f=1000, split='validation', size=1, start_idx=2)
test_dataset = test_dataset.map(prepare).batch(1)
sourcemap, labels = next(iter(test_dataset))

prediction = model.predict(sourcemap, verbose=0)[0]
sourcemap = sourcemap.numpy().squeeze()
extent = dataset.config.grid.extent
loc = labels[0]

Lm = ac.L_p(sourcemap).T
fig, ax = plt.subplots()
im = ax.imshow(Lm, origin='lower', vmin=Lm.max() - 15, extent=extent)
ax.plot(prediction[0], prediction[1], 'x', label='prediction')
ax.plot(loc[0], loc[1], 'x', label='true location')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
fig.colorbar(im, ax=ax, label='SPL / dB')
ax.legend()
plt.show()
