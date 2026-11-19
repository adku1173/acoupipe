"""All classes in this module can be used to calculate and provide data.

Purpose of the Pipeline Module
------------------------------

Classes defined in the :code:`pipeline.py` module have the ability to iteratively perform
tasks on the related computational pipeline to build up a dataset.
The results of these tasks are the features (and labels) associated with a specific
sample of the dataset.
Feature creation tasks can be specified by passing callable functions that are evoked at
each iteration of the :code:`BasePipeline`'s :code:`get_data()` generator method.
It is worth noting that such a data generator can also be used directly to feed a machine
learning model without saving the data to file, as common machine learning frameworks,
such as Tensorflow_, offer the possibility to consume data from Python generators.
Control of the state of the sampling process is maintained via the :code:`sampler`
attribute holding a list of :code:`BaseSampler` derived instances.

.. code-block:: python

    import acoular as ac
    from acoupipe.sampler import NumericAttributeSampler
    from acoupipe.pipeline import BasePipeline, DistributedPipeline
    from scipy.stats import norm

    random_var = norm(loc=1.0, scale=0.5)

    n1 = ac.WNoiseGenerator(sample_freq=24000, num_samples=24000 * 5, rms=1.0, seed=1)

    rms_sampler = NumericAttributeSampler(target=[n1], attribute='rms', random_var=random_var)


    def calculate_squared_rms(sampler):
        n1 = sampler[0].target[0]
        return {'rms_sq': n1.rms**2}


    pipeline = BasePipeline(
        numsamples=5,
        seed=(1,),
        sampler=[rms_sampler],
        feature_func=calculate_squared_rms,
    )

    for data in pipeline.get_data():
        print(data)


The example returns the following output:

.. code-block:: bash

    100%|██████████| 5/5 [00:00<00:00, 2250.65it/s]
    {'idx': 1, 'rms_sq': 1.6044704682820132}
    {'idx': 2, 'rms_sq': 0.15967766038023368}
    {'idx': 3, 'rms_sq': 0.8928756755969107}
    {'idx': 4, 'rms_sq': 0.8034323227265465}
    {'idx': 5, 'rms_sq': 1.2420686428150722}
"""

import inspect
import logging
import os
import warnings
import zlib
from functools import wraps
from time import time

from acoupipe.base import BaseSampler, DataGenerator

import numpy as np
import ray
from numpy.random import RandomState, SeedSequence, default_rng
from tqdm import tqdm
from traits.api import Callable, Dict, Either, Instance, Int, Property, Tuple


def seed_key(name):
    """Return a non-negative integer key of a name, e.g. of a sampler or dataset split.

    The key is identical in every Python process, unlike the randomized ``hash()`` of strings.
    """
    return zlib.crc32(name.encode())


def reseed_sampler(sampler, seed, idx):
    """Re-seed each object in the sampler dict for the data sample with index idx.

    The object with key k receives the random stream of ``SeedSequence([*seed, idx, k])``. Streams
    of different keys, indices or seeds are statistically independent, and the stream of an object
    does not depend on the other objects in the dict.
    """
    for k in sampler:
        seed_sequence = SeedSequence([*seed, idx, k])
        if isinstance(sampler[k], BaseSampler):
            sampler[k].random_state = default_rng(seed_sequence)
        elif isinstance(sampler[k], RandomState):
            sampler[k].seed(seed_sequence.generate_state(4))
        elif isinstance(sampler[k], np.random.Generator):
            sampler[k] = np.random.Generator(sampler[k].bit_generator.__class__(seed_sequence))
        else:
            sampler[k].seed = int(seed_sequence.generate_state(1)[0])


def sample_all(sampler):
    """Invoke the :meth:`sample` method of all :class:`BaseSampler` objects in dict order."""
    for value in sampler.values():
        if isinstance(value, BaseSampler):
            value.sample()


# Without the use of this decorator factory (wraps), the name of the
# function 'f' would have been 'wrap', and the docstring of the original f() would have been lost.
def log_execution_time(f):
    """Log execution time during feature calculation."""

    @wraps(f)
    def wrap(self, *args, **kw):
        self.logger.info(f'id {self._idx}: start task.')
        start = time()
        result = f(self, *args, **kw)
        end = time()
        self.logger.info(f'id {self._idx}: finished task.')
        # self.logger.info(f"{f.__name__} args:[{args}] took: {end-start:.32f} sec")
        self.logger.info(f'id {self._idx}: executing task took: {end - start:.32f} sec')
        return result

    return wrap


class BasePipeline(DataGenerator):
    """Control the random process and iteratively extract and pass a specified amount of data.

    This class can be used to calculate data (extract features)
    by assigning a name and a callable function to :attr:`feature_func`.
    Furthermore this class automatically controles the sampling of instances
    of type :class:`BaseSampler` specified to the :attr:`sampler` list.
    Before each data sample, all objects in :attr:`sampler` are re-seeded from :attr:`seed`, the
    sample index and their key (see :func:`reseed_sampler`).
    """

    #: a dictionary with instances of :class:`~acoupipe.base.BaseSampler` derived classes as values
    #: and non-negative integers as keys. The samplers are invoked in dict order. The key
    #: identifies the random stream of a sampler, so that the samples of a sampler do not change
    #: if other samplers are added or removed.
    #: Alternatively, the dict can contain objects of type
    #: :class:`numpy.random._generator.Generator` or :class:`numpy.random.RandomState`
    #: for control reasons
    #: (e.g. :code:`sampler = {0 : rms_sampler, 1 : numpy.random.RandomState(1)}`).
    #: The trait also accepts a list of samples
    #: (e.g. :code:`sampler = [rms_sampler, numpy.random.RandomState(1)]`).
    #: which is then converted to a dictionary with the indices of the list as keys.
    sampler = Property(
        desc='Dictionary with instances of BaseSampler derived classes as values and their random stream keys as keys',
    )

    #: feature method for the extraction/generation of features and labels.
    #: one can either pass a callable
    #: (e.g. `feature_func = `lambda sampler: {"feature_name" : sampler.target}`).
    #: Note that the callable must accept a list of :class:`acoupipe.base.BaseSampler`
    #: objects as first argument.
    #: Alternatively, if further arguments are necessary, one can pass a tuple containing the
    #: callable and their arguments (e.g.: `feature_func = (some_func, arg1, arg2, ...)}`).
    feature_func = Either(Callable, Tuple, desc='feature method for the extraction/generation of features and labels')

    #: deprecated alias of :attr:`feature_func`.
    features = Property(desc='deprecated alias of feature_func')

    #: tuple of non-negative integers that, together with the sample index and the sampler key,
    #: seeds the samplers of each data sample, e.g. ``(dataset_seed, split_key)``.
    seed = Tuple(desc='non-negative integers seeding the samplers together with sample index and sampler key')

    #: number of samples to calculate by :meth:`get_data()`.
    numsamples = Int(0, desc='number of data samples to calculate')

    #: logger instance to log calculation times for each data sample
    logger = Property(desc='Logger instance to log timing statistics')

    _idx = Int(0, desc='Internal running index')

    _logger = Instance(logging.Logger, desc='Internal logger instance')

    _sampler = Dict(key_trait=Int)

    def _get_logger(self):
        if self._logger is None:
            self._logger = self._get_default_logger()
        return self._logger

    def _set_logger(self, logger):
        self._logger = logger

    def _get_default_logger(self):
        """Set up standard logging to stdout, stderr."""
        logger = logging.getLogger(__name__)
        logger.propagate = False  # don't propagate to the root logger!
        # add standard out handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter('%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d %(message)s', datefmt='%H:%M:%S'),
        )
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stream_handler)
        return logger

    @log_execution_time
    def _extract_features(self):
        """Calculate features."""
        if callable(self.feature_func):
            return self.feature_func(self.sampler)
        if isinstance(self.feature_func, tuple):
            return self.feature_func[0](self.sampler, *list(self.feature_func[1:]))
        if self.feature_func is None:
            return {}
        msg = 'feature_func attribute must be a callable or a tuple containing a callable and its arguments!'
        raise ValueError(msg)

    def _get_sampler(self):
        return self._sampler

    def _set_sampler(self, sampler):
        if isinstance(sampler, list):
            self._sampler = dict(enumerate(sampler))
        else:
            self._sampler = sampler

    def _get_features(self):
        """Return :attr:`feature_func` (deprecated alias)."""
        warnings.warn(
            'BasePipeline.features is deprecated, use BasePipeline.feature_func instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.feature_func

    def _set_features(self, feature_func):
        """Set :attr:`feature_func` (deprecated alias)."""
        warnings.warn(
            'BasePipeline.features is deprecated, use BasePipeline.feature_func instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.feature_func = feature_func

    def _validate_feature_func(self):
        if not callable(self.feature_func):
            if not isinstance(self.feature_func, tuple):
                msg = 'feature_func attribute must be a callable or a tuple containing a callable and its arguments!'
                raise ValueError(msg)
            nargs = len(inspect.signature(self.feature_func[0]).parameters)
            if nargs == 0:
                msg = (
                    'feature functions at must accept at least one argument in order to pass a list of sampler objects!'
                    ' E.g. lambda sampler: {...}'
                )
                raise ValueError(msg)
            if nargs > 1 and len(self.feature_func[1:]) == nargs:
                msg = (
                    'Number of arguments of feature function matches the number of arguments in the tuple!'
                    ' An additional argument is missing to pass the sampler objects!'
                    ' E.g. lambda sampler, arg1, arg2: {...}'
                )
                raise ValueError(
                    msg,
                )
        else:
            nargs = len(inspect.signature(self.feature_func).parameters)
            if nargs == 0:
                msg = (
                    'feature functions at must accept at least one argument in order to pass a list of sampler objects!'
                    ' E.g. lambda sampler: {...}'
                )
                raise ValueError(msg)

    def get_data(self, progress_bar=True, start_idx=1):
        """Provide the extracted features and indices.

        Parameters
        ----------
        progress_bar : bool, optional
            if True, a progress bar is displayed, by default True
        start_idx : int, optional
            the index of the first data sample to be calculated, by default 1

        Yields
        ------
        dict
            a sample of the dataset containing the extracted feature data and index
        """
        self._validate_feature_func()
        pbar = tqdm(total=self.numsamples, colour='#1f77b4', disable=(not progress_bar))
        for idx in range(start_idx, start_idx + self.numsamples):
            self._idx = idx
            reseed_sampler(self.sampler, self.seed, idx)
            sample_all(self.sampler)
            data = {'idx': idx}
            data.update(self._extract_features())
            yield data
            pbar.update(1)
        pbar.close()


@ray.remote
class SamplerActor:
    """Actor class to sample data."""

    def __init__(self, state):
        # IMPORTANT: do NOT import gpuRIR before this point anywhere.
        gpu_ids = ray.get_gpu_ids()  # e.g. [3]
        logging.error(f'ray.get_gpu_ids() = {gpu_ids}, CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}')

        self.sampler, self.feature_func, self.seed = state

    def sample(self, idx):
        """Re-seed and invoke all :class:`BaseSampler` objects of the data sample with index idx."""
        reseed_sampler(self.sampler, self.seed, idx)
        sample_all(self.sampler)
        return self.sampler

    def extract_features(self, indices, times, *feature_args):
        """Remote calculation of all features for each sample index in indices."""
        times[1] = time()
        data = [self.feature_func(self.sample(idx), *feature_args) for idx in indices]
        times[2] = time()
        return (data, times, os.getpid())

    def exit(self):
        ray.actor.exit_actor()


class ActorHandler:
    def __init__(self, numworkers, sampler, feature_func, seed, remote_args=None):
        # Samplers and feature functions may share objects, e.g. a ParameterSet that samplers
        # modify and features read. Ray serializes each argument separately, so both are
        # stored as one object to preserve shared references within every actor.
        state = ray.put((sampler, feature_func, seed))
        self.actors = [SamplerActor.options(**(remote_args or {})).remote(state) for _ in range(numworkers)]

    def __enter__(self):
        return self.actors

    def __exit__(self, type, value, traceback):  # noqa A002
        for actor in self.actors:
            actor.exit.remote()


class DistributedPipeline(BasePipeline):
    """Class to calculate data (extract features) in parallel to build large datasets.

    This class can be used to calculate data (extract various features)
    by assigning a name and a callable function to :attr:`feature_func`.
    Furthermore this class automatically controles the sampling of instances
    of type :class:`BaseSampler` specified to the :attr:`sampler` list.
    Each worker re-seeds the samplers from :attr:`seed`, the sample index and the sampler key, so
    that the samples do not depend on the number of workers.

    Each remote task calculates up to :attr:`samples_per_task` data samples before returning them.
    Since the samples of a task are serialized together, feature functions must not return objects
    that are modified in place by subsequent sampling.

    .. warning::

        Set ``NUMBA_NUM_THREADS=1``, ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1`` and
        ``MKL_NUM_THREADS=1`` before starting Python or the Ray cluster. Otherwise, every worker
        starts as many threads as there are CPU cores, and the workers slow each other down.
    """

    #: number of workers to be used for parallel calculation (usually number of CPUs).
    #: each worker is associated with a stateless task.
    numworkers = Int(1, desc='number of tasks to be performed in parallel (usually number of CPUs)')

    #: maximum number of data samples calculated per remote task. Larger values reduce the
    #: scheduling overhead per sample. To keep all workers busy, a task holds at most
    #: ceil(remaining samples / workers) samples.
    samples_per_task = Int(16, desc='maximum number of data samples calculated per remote task')

    #: additional arguments to be passed to ray remote tasks, e.g. num_gpus for GPU
    #: resource allocation (e.g. num_gpus=0.25 for 1 GPU per 4 tasks)
    remote_args = Dict()

    def _log_execution_time(self, task_index, times, pid):
        self.logger.info(f'id {task_index} on pid {pid}: scheduling task took: {times[1] - times[0]:.32f} sec')
        self.logger.info(f'id {task_index} on pid {pid}: executing task took: {times[2] - times[1]:.32f} sec')
        self.logger.info(f'id {task_index} on pid {pid}: retrieving result took: {times[3] - times[2]:.32f} sec')
        self.logger.info(f'id {task_index} on pid {pid}: full time: {times[3] - times[0]:.32f} sec')
        # self.logger.info(f"{f.__name__} args:[{args}] took: {end - start:.4f} sec")

    def _schedule_task(self, actor, task_dict, remaining, nworkers):
        """Schedule the next samples as one remote task of the actor and return their number."""
        num = min(self.samples_per_task, -(-remaining // nworkers))  # keeps all workers busy at the end
        indices = list(range(self._idx + 1, self._idx + 1 + num))
        self._idx += num
        self.logger.info(f'id {indices[0]}: start task with {num} samples.')
        times = [time(), None, None, None]  # (schedule timestamp, execution timestamp, stop timestamp, get timestamp)
        if callable(self.feature_func):
            result_id = actor.extract_features.remote(indices, times)  # calculation is started in new remote task
        elif isinstance(self.feature_func, tuple):
            result_id = actor.extract_features.remote(indices, times, *list(self.feature_func[1:]))
        else:
            msg = 'feature_func attribute must be a callable or a tuple containing a callable and its arguments!'
            raise ValueError(msg)
        task_dict[result_id] = (actor, indices)
        return num

    def get_data(self, progress_bar=True, start_idx=1):
        """Provide the extracted features and indices.

        The calculation of all data samples is performed in parallel and asynchronously.
        In case of specifying more than one worker in the :attr:`numworker` attribute,
        the output of this generator yields non-ordered features/data samples.
        However, the exact order can be recovered via the "idx" item
        provided in the output dictionary.

        Parameters
        ----------
        progress_bar : bool, optional
            if True, a progress bar is displayed, by default True
        start_idx : int, optional
            the index of the first data sample to be calculated, by default 1

        Yields
        ------
        dict
            A sample of the dataset containing the extracted feature data and index
        """
        self._validate_feature_func()
        self._idx = start_idx - 1
        nsamples = self.numsamples
        nworkers = min(nsamples, self.numworkers)
        progress_bar = tqdm(range(nsamples), colour='#1f77b4', disable=(not progress_bar))
        feature_func = self.feature_func if callable(self.feature_func) else self.feature_func[0]
        task_dict = {}
        scheduled = 0
        with ActorHandler(nworkers, self.sampler, feature_func, self.seed, remote_args=self.remote_args) as actors:
            for actor in actors:
                scheduled += self._schedule_task(actor, task_dict, nsamples - scheduled, nworkers)
            while task_dict:
                did = ray.wait(list(task_dict.keys()))[0][0]
                try:
                    results, times, pid = ray.get(did)
                except Exception as exception:
                    self.logger.info(f'task with id {task_dict[did][1][0]} failed with Traceback:', exc_info=True)
                    raise exception
                times[-1] = time()  # add getter time
                actor, indices = task_dict.pop(did)
                self.logger.info(f'id {indices[0]} on pid {pid}: finished task with {len(indices)} samples.')
                self._log_execution_time(indices[0], times, pid)
                if scheduled < nsamples:  # directly schedule the next task
                    scheduled += self._schedule_task(actor, task_dict, nsamples - scheduled, nworkers)
                for data, idx in zip(results, indices, strict=True):
                    data['idx'] = idx
                    yield data
                    progress_bar.update(1)
