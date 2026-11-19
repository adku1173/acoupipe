"""Calculate data samples in parallel as a lazy :class:`ray.data.Dataset`."""

from acoupipe.pipeline import BasePipeline, reseed_sampler, sample_all
from acoupipe.raydata.encodings import split_complex, stack_values

import numpy as np
import ray
from traits.api import Dict, Int


class SampleBatchCalculator:
    """Calculate the features of a batch of sample positions inside a Ray Data actor.

    The constructor receives sampler and feature function as one object so that objects shared
    by both, e.g. a ParameterSet, remain shared after deserialization in the actor.
    """

    def __init__(self, state):
        self.sampler, self.feature_func, self.feature_args, self.seed, self.start_idx = state

    def __call__(self, batch):
        """Calculate the samples of all positions in the batch and stack them per feature."""
        samples = [self.calculate(int(position)) for position in batch['data']]
        return {key: stack_values([sample[key] for sample in samples]) for key in samples[0]}

    def calculate(self, position):
        """Sample and calculate the features of the data sample at the given position."""
        idx = self.start_idx + position
        reseed_sampler(self.sampler, self.seed, idx)
        sample_all(self.sampler)
        data = split_complex(self.feature_func(self.sampler, *self.feature_args))
        data['idx'] = idx
        return data


class RayDataPipeline(BasePipeline):
    """Calculate data in parallel with Ray Data.

    The data samples form a lazy :class:`ray.data.Dataset` (see :meth:`get_dataset`) with one row
    per sample. Each of the :attr:`numworkers` actors calculates up to :attr:`samples_per_task`
    samples per call. Complex features are stored as two real columns ``<name>_real`` and
    ``<name>_imag``, since Arrow has no complex data type. As for
    :class:`~acoupipe.pipeline.DistributedPipeline`, the samples are not ordered; the order can be
    recovered from the "idx" item.

    Requires Ray Data (``ray[data]``).

    .. warning::

        Set ``NUMBA_NUM_THREADS=1``, ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` and
        ``MKL_NUM_THREADS=1`` before starting Python or the Ray cluster. Otherwise, every worker
        starts as many threads as there are CPU cores, and the workers slow each other down.
    """

    #: number of actors calculating features in parallel
    numworkers = Int(1, desc='number of actors calculating features in parallel')

    #: number of data samples calculated per actor call
    samples_per_task = Int(16, desc='number of data samples calculated per actor call')

    #: additional arguments passed to the actors, e.g. num_gpus
    remote_args = Dict()

    def get_dataset(self, start_idx=1):
        """Return a lazy :class:`ray.data.Dataset` with one row per data sample.

        Parameters
        ----------
        start_idx : int, optional
            the index of the first data sample to be calculated, by default 1

        Returns
        -------
        ray.data.Dataset
            dataset with the extracted features and index as columns
        """
        self._validate_feature_func()
        if callable(self.feature_func):
            feature_func, feature_args = self.feature_func, ()
        else:
            feature_func, feature_args = self.feature_func[0], self.feature_func[1:]
        state = (self.sampler, feature_func, feature_args, self.seed, start_idx)
        # The sample positions are created in the driver, one block per actor call. Unlike
        # ray.data.range, this requires no read task, which would need a CPU besides the actors.
        positions = np.arange(self.numsamples)
        blocks = np.split(positions, range(self.samples_per_task, self.numsamples, self.samples_per_task))
        return ray.data.from_numpy(blocks).map_batches(
            SampleBatchCalculator,
            fn_constructor_args=(state,),
            batch_size=self.samples_per_task,
            compute=ray.data.ActorPoolStrategy(size=self.numworkers),
            batch_format='numpy',
            **self.remote_args,
        )

    def get_data(self, progress_bar=True, start_idx=1):
        """Provide the extracted features and indices.

        Parameters
        ----------
        progress_bar : bool, optional
            if True, the Ray Data progress bars are displayed, by default True
        start_idx : int, optional
            the index of the first data sample to be calculated, by default 1

        Yields
        ------
        dict
            A sample of the dataset containing the extracted feature data and index
        """
        ray.data.DataContext.get_current().enable_progress_bars = progress_bar
        yield from self.get_dataset(start_idx).iter_rows()
