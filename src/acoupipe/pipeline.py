"""All classes in this module can be used to calculate and provide data.

Purpose of the Pipeline Module
------------------------------

Classes defined in the :code:`pipeline.py` module have the ability to iteratively perform tasks on the related computational pipeline to build up a dataset.
The results of these tasks are the features (and labels) associated with a specific sample of the dataset.
Feature creation tasks can be specified by passing callable functions that are evoked at each iteration of the :code:`BasePipeline`'s :code:`get_data()` generator method.
It is worth noting that such a data generator can also be used directly to feed a machine learning model without saving the data to file, as common machine learning frameworks, such as Tensorflow_, offer the possibility to consume data from Python generators.
Control of the state of the sampling process is maintained via the :code:`sampler` attribute holding a list of :code:`BaseSampler` derived instances.

.. code-block:: python

    def calculate_csm(powerspectra):
        return powerspectra.csm

    pipeline = acoupipe.BasePipeline(
        sampler=[rms_sampler],
        numsamples = 5,
        features={'csm' : (calculate_csm, ps),}
        )

    data_generator = pipeline.get_data()
"""

import logging
import os
from functools import wraps
from time import time

import ray
from numpy import array
from numpy.random import RandomState, default_rng
from tqdm import tqdm
from traits.api import Callable, Dict, Either, HasPrivateTraits, Instance, Int, Property, Tuple

from acoupipe.sampler import BaseSampler


# Without the use of this decorator factory (wraps), the name of the
# function 'f' would have been 'wrap', and the docstring of the original f() would have been lost.
def log_execution_time(f):
    """Log execution time during feature calculation."""
    @wraps(f)
    def wrap(self, *args, **kw):
        self.logger.info("id %i: start task." %self._idx)
        start = time()
        result = f(self, *args, **kw)
        end = time()
        self.logger.info("id %i: finished task." %self._idx)
        # self.logger.info("%r args:[%r] took: %2.32f sec" % \
        # (f.__name__,args,end-start))
        self.logger.info("id %i: executing task took: %2.32f sec" % \
        (self._idx,end-start))
        return result
    return wrap


class DataGenerator(HasPrivateTraits):
    """Abstract base class that serves as a data generator.

    This class should not be used.
    """

    def get_data(self):
        """Python generator that iteratively yields data set samples as a dictionary.

        Returns
        -------
        Dictionary containing a sample of the data set {feature_name[key],feature[values]}.
        """
        pass



class BasePipeline(DataGenerator):
    """Class to control the random process and iteratively extract and pass a specified amount of data.

    This class can be used to calculate data (extract features)
    by assigning a name and a callable function to :attr:`features`.
    Furthermore this class automatically controles the sampling of instances
    of type :class:`BaseSampler` specified to the :attr:`sampler` list.
    Re-seeding is performed at each iteration if :attr:`random_seeds` are
    given.
    """

    #: a list with instances of :class:`~acoupipe.sampler.BaseSampler` derived classes
    #: alternatively, the list can contain objects of type :class:`numpy.random._generator.Generator` or
    #: :class:`numpy.random.RandomState` for control reasons
    sampler = Dict(key_trait=Int,
        desc="a list with instances of BaseSampler derived classes")

    #: feature method for the extraction/generation of features and labels.
    #: one can either pass a callable (e.g. `features = `lambda sampler: {"feature_name" : sampeler.target}`).
    #: Note that the callable must accept a list of :class:`acoupipe.sampler.BaseSampler` objects as first argument.
    #: Alternatively, if further arguments are necessary, one can pass a tuple containing the
    #: callable and their arguments (e.g.: `features = (some_func, arg1, arg2, ...)}`).
    features = Either(Callable, Tuple,
        desc="feature method for the extraction/generation of features and labels")

    #: a list of `range(seeds)` associated with sampler objects in :attr:`sampler`.
    #: A new seed will be collected from each range object during an evaluation of the :meth:`get_data()` generator.
    #: This seed is used to initialize an instance of :class:`numpy.random._generator.Generator` which is passed to
    #: the :attr:`random_state` of the samplers in :attr:`sampler`. If not given, :meth:`get_data()` relies on
    #: :attr:`numsamples`.
    random_seeds = Dict(key_trait=Int, value_trait=range,
        desc="List of seeds associated with sampler objects")

    #: number of samples to calculate by :meth:`get_data()`.
    #: Will be superseded by the :attr:`random_seeds` attribute if specified.
    numsamples = Int(0,
        desc="number of data samples to calculate. Will be superseded by the random_seeds attribute if specified")

    _idx = Int(0,
        desc="Internal running index")

    _seeds = Dict(key_trait=Int, value_trait=Int,
        desc="Internal running seeds")

    #: logger instance to log calculation times for each data sample
    logger = Property(desc="Logger instance to log timing statistics")

    _logger = Instance(logging.Logger, desc="Internal logger instance")

    def _get_logger(self):
        if self._logger is None:
            self._logger = self._get_default_logger()
        return self._logger

    def _set_logger(self, logger):
        self._logger = logger

    def _get_default_logger(self):
        """Set up standard logging to stdout, stderr."""
        logger = logging.getLogger(__name__)
        logger.propagate = False # don't propagate to the root logger!
        # add standard out handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d %(message)s",
                datefmt="%H:%M:%S"))
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(stream_handler)
        return logger

    def _validate_random_seeds(self):
        """Validate specified random seeds."""
        if self.random_seeds:
            if len(self.random_seeds.keys()) != len(self.sampler.keys()):
                raise ValueError("Number of given range objects in random_seeds"\
                                  "and number of sampler objects need to be equal!")
            if len(set(list(map(len,self.random_seeds.values())))) != 1:
                raise ValueError("Length of range objects in random_seeds"\
                                 "list must be equal!")
    @log_execution_time
    def _extract_features(self):
        """Calculate features."""
        if callable(self.features):
            return self.features(self.sampler)
        else:
            return self.features[0](self.sampler, *list(self.features[1:]))

    def _update_sample_index_and_seeds(self, seed_iter=None):
        """Update seeds and running index of associated with the current data sample of the dataset."""
        self._idx += 1
        if self.random_seeds:
            self._seeds = {k : next(seed_iter[k]) for k in seed_iter.keys()}
            for k in self.sampler.keys():

                if isinstance(self.sampler[k],BaseSampler):
                    self.sampler[k].random_state = default_rng(self._seeds[k])
                    #self.logger.error(f"update {self.sampler[k].__class__.__name__}, state: {self.sampler[k].random_state.__getstate__()}")
                elif isinstance(self.sampler[k], RandomState):
                    self.sampler[k].seed(self._seeds[k])
                else:
                    self.sampler[k].seed = self._seeds[k]

    def get_data(self, progress_bar=True, start_idx=1):
        """Provide the extracted features, sampler seeds and indices.

        Parameters
        ----------
        progress_bar : bool, optional
            if True, a progress bar is displayed, by default True
        start_idx : int, optional
            the index of the first data sample to be calculated, by default 1

        Yields
        ------
        dict
            a sample of the dataset containing the extracted feature data, seeds, and index
        """
        self._idx = (start_idx-1)
        if self.random_seeds:
            self._validate_random_seeds()
            seed_iter = {k : iter(v) for k,v in self.random_seeds.items()}
            nsamples = len(list(self.random_seeds.values())[0])
        else:
            seed_iter = None
            nsamples = self.numsamples
        sampler_order = list(self.sampler.keys())
        sampler_order.sort()
        pbar = tqdm(iterable=range(start_idx, nsamples+start_idx),total=nsamples,colour="#1f77b4",disable=(not progress_bar),)
        for _ in range(nsamples):
            self._update_sample_index_and_seeds(seed_iter)
            for i in sampler_order:
                if isinstance(self.sampler[i], BaseSampler):
                    self.sampler[i].sample()
            data = {"idx" : self._idx, "seeds": array(list(self._seeds.items()))}
            data.update(self._extract_features())
            yield data
            pbar.update(1)
        pbar.close()


@ray.remote
class SamplerActor(object):
    """Actor class to sample data."""

    def __init__(self, sampler, feature_func):
        self.sampler = sampler
        self.feature_func = feature_func
        self.sampler_order = list(self.sampler.keys())
        self.sampler_order.sort()

    def sample(self, seeds):
        """Invocation of the :meth:`sample` function of one or more :class:`BaseSampler` instances."""
        self.set_new_seed(seeds)
        for k in self.sampler_order:
            if isinstance(self.sampler[k], BaseSampler):
                self.sampler[k].sample()
        return self.sampler

    def extract_features(self, seeds, times, *feature_args):
        """Remote calculation of all features."""
        times[1] = time()
        data = self.feature_func(self.sample(seeds), *feature_args)
        times[2] = time()
        return (data, times, os.getpid())

    def set_new_seed(self, seeds=None):
        """Re-seeds :class:`BaseSampler` instances specified in :attr:`sampler` dict."""
        if seeds:
            for k in self.sampler.keys():
                if isinstance(self.sampler[k],BaseSampler):
                    self.sampler[k].random_state = default_rng(seeds[k])
                elif isinstance(self.sampler[k], RandomState):
                    self.sampler[k].seed(seeds[k])
                else:
                    self.sampler[k].seed = seeds[k]

    def exit(self):
        ray.actor.exit_actor()

class ActorHandler(object):
    def __init__(self, numworkers, sampler, feature_func):
        self.actors = [
            SamplerActor.remote(
                sampler=sampler, feature_func=feature_func
                ) for _ in range(numworkers)]
    def __enter__(self):
        return self.actors
    def __exit__(self, type, value, traceback):
        for actor in self.actors:
            actor.exit.remote()


class DistributedPipeline(BasePipeline):
    """Class to calculate data (extract features) in parallel to build large datasets.

    This class can be used to calculate data (extract various features)
    by assigning a name and a callable function to :attr:`features`.
    Furthermore this class automatically controles the sampling of instances
    of type :class:`BaseSampler` specified to the :attr:`sampler` list.
    Re-seeding is performed at each iteration if :attr:`random_seeds` are
    given.
    """

    #: number of workers to be used for parallel calculation (usually number of CPUs).
    #: each worker is associated with a stateless task.
    numworkers = Int(1,
        desc="number of tasks to be performed in parallel (usually number of CPUs)")

    def _log_execution_time(self,task_index,times,pid):
        self.logger.info("id %i on pid %i: scheduling task took: %2.32f sec" % \
        (task_index, pid, times[1]-times[0]))
        self.logger.info("id %i on pid %i: executing task took: %2.32f sec" % \
        (task_index, pid, times[2]-times[1]))
        self.logger.info("id %i on pid %i: retrieving result took: %2.32f sec" % \
        (task_index,pid, times[3]-times[2]))
        self.logger.info("id %i on pid %i: full time: %2.32f sec" % \
        (task_index,pid, times[3]-times[0]))
        # self.logger.info("%i args:[%r] took: %2.4f sec" % \
        # (f.__name__,args,end-start))

    def _sample_and_schedule_task(self,actor,task_dict):
        self.logger.info("id %i: start task." %self._idx)
        times = [time(), None, None, None] # (schedule timestamp, execution timestamp, stop timestamp, get timestamp)
        if callable(self.features):
            result_id = actor.extract_features.remote(self._seeds, times) # calculation is started in new remote task
        else:
            result_id = actor.extract_features.remote(self._seeds, times, *list(self.features[1:]))
        task_dict[result_id] = (actor, {"idx" : self._idx, "seeds": array(list(self._seeds.items()))})  # add index, and seeds

    def _update_sample_index_and_seeds(self, seed_iter=None):
        """Update seeds and running index of associated with the current data sample of the dataset."""
        self._idx += 1
        if self.random_seeds:
            self._seeds = {k : next(seed_iter[k]) for k in seed_iter.keys()}

    def get_data(self, progress_bar=True, start_idx=1):
        """Provide the extracted features, sampler seeds and indices.

        The calculation of all data samples is performed in parallel and asynchronously.
        In case of specifying more than one worker in the :attr:`numworker` attribute,
        the output of this generator yields non-ordered features/data samples.
        However, the exact order can be recovered via the "idx" item (or "seeds" item)
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
            a sample of the dataset containing the extracted feature data, seeds, and index
        """
        self._idx = (start_idx-1)
        if self.random_seeds:
            self._validate_random_seeds()
            seed_iter = {k : iter(v) for k,v in self.random_seeds.items()}
            nsamples = len(list(self.random_seeds.values())[0])
        else:
            seed_iter = None
            nsamples = self.numsamples
        nworkers = min(nsamples,self.numworkers)
        progress_bar = tqdm(range(nsamples),colour="#1f77b4",disable=(not progress_bar))
        if callable(self.features):
            feature_func = self.features
        else:
            feature_func = self.features[0]
        task_dict = {}
        finished_tasks = 0
        with ActorHandler(nworkers, self.sampler, feature_func) as actors:
            for actor in actors:
                self._update_sample_index_and_seeds(seed_iter)
                self._sample_and_schedule_task(actor,task_dict)
            while finished_tasks < nsamples:
                done_ids, pending_ids = ray.wait(list(task_dict.keys()))
                if done_ids:
                    id = done_ids[0]
                    finished_tasks += 1
                    try:
                        data, times, pid = ray.get(id)
                    except Exception as exception:
                        self.logger.info(
                            f"task with id {task_dict[id]} failed with Traceback:",
                            exc_info=True)
                        raise exception
                    times[-1] = time() # add getter time
                    actor, new_data = task_dict.pop(id)
                    data.update(new_data) # add the remaining task_dict items to the data dict
                    self.logger.info(f"id {data['idx']} on pid {pid}: finished task.")
                    self._log_execution_time(data["idx"], times, pid)
                    if (nsamples + start_idx - 1 - self._idx) > 0: # directly _schedule next task
                        self._update_sample_index_and_seeds(seed_iter)
                        self._sample_and_schedule_task(actor,task_dict)
                    yield data
                    progress_bar.update(1)
