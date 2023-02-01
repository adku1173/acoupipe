"""All classes in this module can be used to calculate and provide data.

.. autosummary::
    :toctree: generated/

    DataGenerator
    BasePipeline
    DistributedPipeline

"""

import logging
import os
from functools import wraps
from time import time

import ray
from numpy import array
from numpy.random import RandomState, default_rng
from tqdm import tqdm
from traits.api import Callable, Dict, Either, HasPrivateTraits, Int, Trait, Tuple

from acoupipe.sampler import BaseSampler


# Without the use of this decorator factory (wraps), the name of the 
# function 'f' would have been 'wrap', and the docstring of the original f() would have been lost.
def log_execution_time(f):
    """Decorator to log execution time during feature calculation."""
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

    def __init__(self,*args,**kwargs):
        HasPrivateTraits.__init__(self,*args,**kwargs)
        if "logger" not in kwargs.keys():
            self._setup_default_logger() # define logger formatting if not specified otherwise      

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
    logger = Trait(logging.getLogger(__name__),
        desc="Logger instance to log timing statistics")

    def _setup_default_logger(self):
        """standard logging to stdout, stderr."""
        #print(f"setup default logger is called by {self}")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d-%(message)s",
                datefmt="%Y-%m-%d,%H:%M:%S"))
        self.logger.addHandler(stream_handler)
        self.logger.propagate = True # don't propagate to the root logger!     

    def _validate_random_seeds(self):
        """validation of specified random seeds."""
        if self.random_seeds:
            if len(self.random_seeds.keys()) != len(self.sampler.keys()):
                raise ValueError("Number of given range objects in random_seeds"\
                                  "and number of sampler objects need to be equal!")
            if len(set(list(map(len,self.random_seeds.values())))) != 1:
                raise ValueError("Length of range objects in random_seeds"\
                                 "list must be equal!")

    @log_execution_time   
    def _extract_features(self):
        """calculation of all features."""
        if callable(self.features):
            return self.features(self.sampler)
        else:
            return self.features[0](self.sampler, *list(self.features[1:]))
           
    def _update_sample_index_and_seeds(self, seed_iter=None):
        """updates seeds and running index of associated with the current data sample of the dataset."""
        self._idx += 1
        if self.random_seeds: 
            self._seeds = {k : next(seed_iter[k]) for k in seed_iter.keys()}
            for k in self.sampler.keys():
                if isinstance(self.sampler[k],BaseSampler):
                    self.sampler[k].random_state = default_rng(self._seeds[k]) 
                elif isinstance(self.sampler[k], RandomState):
                    self.sampler[k].seed(self._seeds[k])
                else:
                    self.sampler[k].seed = self._seeds[k]
                
    def get_data(self, progress_bar=True):
        """provides the extracted features, sampler seeds and indices.

        Parameters
        ----------
        progress_bar : bool, optional
            if True, a progress bar is displayed, by default True

        Yields
        ------
        dict
            a sample of the dataset containing the extracted feature data, seeds, and index
        """
        if self.random_seeds: 
            self._validate_random_seeds()
            seed_iter = {k : iter(v) for k,v in self.random_seeds.items()}
            nsamples = len(list(self.random_seeds.values())[0])
        else:
            seed_iter = None
            nsamples = self.numsamples
        sampler_order = list(self.sampler.keys())
        sampler_order.sort()
        self._idx = 0
        for _ in tqdm(range(nsamples),colour="#1f77b4", disable=(not progress_bar)):
            self._update_sample_index_and_seeds(seed_iter)
            for i in sampler_order:
                if isinstance(self.sampler[i], BaseSampler):
                    self.sampler[i].sample()
            data = {"idx" : self._idx, "seeds": array(list(self._seeds.items()))}
            data.update(self._extract_features())
            yield data


# logging timing statistics has to be performed differently in this class, since we don't want the logger
# to get pickled!
# example for logging during multiprocessing: https://fanchenbao.medium.com/python3-logging-with-multiprocessing-f51f460b8778

@ray.remote # pseudo calc function that should run asynchronously
def _extract_features(sampler_ref, feature_func, times, *args):
    """Remote calculation of all features."""
    times[1] = time()
    data = feature_func(sampler_ref, *args)
    times[2] = time()
    return (data, times, os.getpid())


@ray.remote
class SamplerActor(object):
    """Actor class to sample data."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.sampler_order = list(self.sampler.keys())
        self.sampler_order.sort()

    def sample(self, seeds):
        """Invocation of the :meth:`sample` function of one or more :class:`BaseSampler` instances."""
        self.set_new_seed(seeds)
        for k in self.sampler_order:
            if isinstance(self.sampler[k], BaseSampler):
                self.sampler[k].sample()
        return self.sampler

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

    def _schedule(self, sampler_ref, task_dict):
        """schedules the calculation of a new data sample and adds the sample index and start time to a task dictionary."""
        times = [time(), None, None, None] # (schedule timestamp, execution timestamp, stop timestamp, get timestamp)
        if callable(self.features): 
            result_id = _extract_features.remote(sampler_ref, self.features, times) # calculation is started in new remote task 
        else:
            result_id = _extract_features.remote(sampler_ref, self.features[0], times, *list(self.features[1:]))  
        task_dict[result_id] = {"idx" : self._idx, "seeds": array(list(self._seeds.items()))}  # add index, and seeds
        
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

    def _sample_and_schedule_task(self,sampler_actor,task_dict):
        sampler_ref = sampler_actor.sample.remote(self._seeds)
        self.logger.info("id %i: start task." %self._idx)
        self._schedule(sampler_ref,task_dict)

    def _update_sample_index_and_seeds(self, seed_iter=None):
        """updates seeds and running index of associated with the current data sample of the dataset."""
        self._idx += 1
        if self.random_seeds: 
            self._seeds = {k : next(seed_iter[k]) for k in seed_iter.keys()}

    def get_data(self, progress_bar=True):
        """Provides the extracted features, sampler seeds and indices.

        The calculation of all data samples is performed in parallel and asynchronously.
        In case of specifying more than one worker in the :attr:`numworker` attribute, 
        the output of this generator yields non-ordered features/data samples. 
        However, the exact order can be recovered via the "idx" item (or "seeds" item) 
        provided in the output dictionary. 

        Parameters
        ----------
        progress_bar : bool, optional
            if True, a progress bar is displayed, by default True

        Yields
        ------
        dict
            a sample of the dataset containing the extracted feature data, seeds, and index
        """
        sampler_actors = [SamplerActor.remote(sampler=self.sampler) for _ in range(self.numworkers)]
        if self.random_seeds: 
            self._validate_random_seeds()
            seed_iter = {k : iter(v) for k,v in self.random_seeds.items()}
            nsamples = len(list(self.random_seeds.values())[0])
        else:
            seed_iter = None
            nsamples = self.numsamples
        progress_bar = tqdm(range(nsamples),colour="#1f77b4",disable=(not progress_bar))
        task_dict = {}
        finished_tasks = 0
        actor_idx = 0
        for _ in range(min(nsamples,self.numworkers)): 
            self._update_sample_index_and_seeds(seed_iter)
            self._sample_and_schedule_task(sampler_actors[actor_idx],task_dict)
            actor_idx = (actor_idx + 1) % self.numworkers
        while finished_tasks < nsamples: 
            done_ids, pending_ids = ray.wait(list(task_dict.keys()))
            if done_ids:
                id = done_ids[0]
                finished_tasks += 1
                try:
                    data, times, pid = ray.get(id)
                except Exception as exception:
                    self.logger.info("task with id %s failed with Traceback:" %task_dict[id], exc_info=True)
                    raise exception
                times[-1] = time() # add getter time
                data.update(task_dict.pop(id)) # add the remaining task_dict items to the data dict
                #del id
                self.logger.info("id %i on pid %i: finished task." %(data["idx"],pid))
                self._log_execution_time(data["idx"], times, pid)
                if (nsamples - self._idx) > 0: # directly _schedule next task
                    self._update_sample_index_and_seeds(seed_iter)
                    self._sample_and_schedule_task(sampler_actors[0],task_dict)
                    actor_idx = (actor_idx + 1) % self.numworkers
                progress_bar.update()
                yield data
