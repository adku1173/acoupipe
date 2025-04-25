from abc import abstractmethod
from functools import partial
from typing import Dict, Type

import acoular as ac
import numpy as np
from numpy.random import default_rng
from pydantic import BaseModel, Field
from scipy.stats import norm, poisson
from traits.api import ABCHasStrictTraits, Any, Bool, CArray, Float, Instance, Int, Tuple, Union

import acoupipe.sampler as sp
from acoupipe.new_datasets.config import DEFAULT_MICS
from acoupipe.new_datasets.utils import add_split, get_seed


def sample_signal_seed(rng):
    return int(rng.uniform(1,1e9))

def sample_rms(nsources, rng):
    """Draw sources' squared rms pressures from Rayleigh distribution."""
    return np.sqrt(rng.rayleigh(5,nsources))

def sample_mic_noise_variance(rng): # TODO: sample SNR!
    """Draw microphone noise variance, uniform distribution."""
    return rng.uniform(10e-6,0.1)

def sample_signal_length(rng):
    """Draw signal length, uniform distribution."""
    return rng.uniform(1,10)

def nsources_sampler_fn(data, sampler, sampler_id):
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    data["nsources"] = sampler.target.pop()
    return data

def mic_pos_sampler_fn(data, sampler, sampler_id):
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    data["noisy_mic_pos"] = sampler.target.pos
    data["mic_pos"] = sampler.mpos_init
    return data

def signal_seed_sampler_fn(data, sampler, sampler_id):
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    data["signal_seeds"] = np.arange(data["nsources"]) + sampler.target
    return data

def rms_sampler_fn(data, sampler, sampler_id):
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    # get microphone closest to the array center
    mic_pos = data.get("noisy_mic_pos", data["mic_pos"])
    mg = ac.MicGeom(pos_total=mic_pos)
    center = mg.center
    mic_pos = mic_pos[:, np.argmin(np.linalg.norm(mic_pos-center[:,np.newaxis], axis=0))]
    r0 = np.linalg.norm(mic_pos[:,np.newaxis]-data["loc"],axis=0)
    data["rms"] = sampler.target[:data["nsources"]]*r0
    return data

def loc_sampler_fn(data, sampler, sampler_id):
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.nsources = data["nsources"]
    sampler.sample()
    data["loc"] = sampler.target
    return data

def mic_noise_variance_sampler_fn(data, sampler, sampler_id):
    """Draw microphone noise variance, uniform distribution."""
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    data["mic_noise_variance"] = sampler.target
    return data

def signal_length_sampler_fn(data, sampler, sampler_id):
    """Draw signal length, uniform distribution."""
    sampler.random_state = default_rng(get_seed(data["idx"], sampler_id, data["split"]))
    sampler.sample()
    data["signal_length"] = sampler.target
    return data


class MonteCarloBase(ABCHasStrictTraits):

    @abstractmethod
    def build_map_fn(self, split="training"):
        # build the map function for the dataset
        # this is a stateless function that can be used in the map
        # function of the dataset
        split_fn = partial(add_split, split=split)
        def map_fn(data):
            data = split_fn(data)
            # add some sampled parameters to the data
            return data
        return map_fn

    def add_ray_map(self, ray_dataset, split="training", **kwargs):
        return ray_dataset.map(self.build_map_fn(split=split),
                               runtime_env={"env_vars": {"NUMBA_NUM_THREADS": "1"}},
                               **kwargs)



class MonteCarloSynthetic(MonteCarloBase):

    # random number of sources
    random_nsources = Bool(True)
    max_nsources = Int(10, desc="maximum number of sources")
    min_nsources = Int(1, desc="minimum number of sources")

    # positional noise sampling
    mic_pos_noise = Bool(True)
    mic_pos = CArray(shape=(3, None), dtype=float, value=DEFAULT_MICS.pos)
    mic_pos_noise_random_var = Any(default_value=norm(loc=0, scale=0.001))
    mic_pos_noise_ddir = CArray(shape=(3, 1), dtype=float, value=np.array([[1.0], [1.0], [0]]))

    # signal rms sampling
    rms_random_fn = Any(default_value=sample_rms)

    # source location sampling
    source_grid = Union(None, Instance(ac.Grid), CArray(shape=(3, None), dtype=float))
    snap_to_grid = Bool(False, desc="snap source locations to grid")
    source_loc_x_random_var = Any(None)
    source_loc_y_random_var = Any(None)
    source_loc_z_random_var = Any(None)
    source_loc_x_bounds = Union(None, Tuple)
    source_loc_y_bounds = Union(None, Tuple)
    source_loc_z_bounds = Union(None, Tuple)

    # mic noise variance sampling
    mic_sig_noise = Bool(True)
    mic_noise_random_fn = Any(default_value=sample_mic_noise_variance)

    # signal length sampling
    signal_length = Float(5, desc="length of the source signals in seconds")
    random_signal_length = Bool(False)
    signal_length_random_fn = Any(default_value=sample_signal_length)

    # private
    _nsources_sampler = Instance(sp.NumericAttributeSampler)
    _micgeom_sampler = Instance(sp.MicGeomSampler)
    _signal_seed_sampler = Instance(sp.ContainerSampler)
    _rms_sampler = Instance(sp.ContainerSampler)
    _location_sampler = Instance(sp.LocationSampler)
    _mic_noise_sampler = Instance(sp.ContainerSampler)
    _signal_length_sampler = Instance(sp.ContainerSampler)

    def create_nsources_sampler(self):
        self._nsources_sampler = sp.NumericAttributeSampler(random_var = poisson(mu=3, loc=1),
            filter=lambda x: (x <= self.max_nsources) and (x >= self.min_nsources))

    def create_micgeom_sampler(self):
        self._micgeom_sampler = sp.MicGeomSampler(
                random_var=self.mic_pos_noise_random_var, ddir=self.mic_pos_noise_ddir,
                mpos_init=self.mic_pos)

    def create_signal_seed_sampler(self):
        self._signal_seed_sampler = sp.ContainerSampler(random_func = sample_signal_seed)

    def create_rms_sampler(self):
        self._rms_sampler = sp.ContainerSampler(
            random_func = partial(self.rms_random_fn, self.max_nsources))

    def create_location_sampler(self):
        mics = ac.MicGeom(pos_total=self.mic_pos)
        ap = mics.aperture
        if self.source_loc_x_random_var is None:
            self.source_loc_x_random_var = norm(loc=0, scale=0.1688*ap)

        if self.source_loc_y_random_var is None:
            self.source_loc_y_random_var = norm(loc=0, scale=0.1688*ap)

        if self.source_loc_z_random_var is None:
            self.source_loc_z_random_var = norm(loc=0.5*ap, scale=0)

        if self.source_loc_x_bounds is None:
            self.source_loc_x_bounds = (-0.5*ap, 0.5*ap)

        if self.source_loc_y_bounds is None:
            self.source_loc_y_bounds = (-0.5*ap, 0.5*ap)

        if self.source_loc_z_bounds is None:
            self.source_loc_z_bounds = (0.5*ap, 0.5*ap)

        location_sampler = sp.LocationSampler(
            random_var = (self.source_loc_x_random_var,
                          self.source_loc_y_random_var,
                          self.source_loc_z_random_var),
            x_bounds = self.source_loc_x_bounds,
            y_bounds = self.source_loc_y_bounds,
            z_bounds = self.source_loc_z_bounds,
            nsources = self.max_nsources)
        if self.snap_to_grid:
            if self.source_grid is None:
                raise ValueError("source_grid must be given in the constructor if snap_to_grid is True")
            if not isinstance(self.source_grid, ac.Grid):
                self.source_grid = ac.ImportGrid(pos=self.source_grid)
            location_sampler.grid = self.source_grid
        self._location_sampler = location_sampler

    def create_mic_noise_sampler(self):
        self._mic_noise_sampler = sp.ContainerSampler(
            random_func = self.mic_noise_random_fn)

    def create_signal_length_sampler(self):
        self._signal_length_sampler = sp.ContainerSampler(
            random_func = self.signal_length_random_fn)

    def _build_nsources_map_fn(self):
        if self.random_nsources and self.max_nsources != self.min_nsources:
            self.create_nsources_sampler()
            return partial(nsources_sampler_fn, sampler=self._nsources_sampler, sampler_id=0)
        else:
            def add_nsources(data):
                data["nsources"] = self.max_nsources
                return data
            return add_nsources

    def _build_mic_pos_map_fn(self):
        if self.mic_pos_noise:
            self.create_micgeom_sampler()
            return partial(mic_pos_sampler_fn, sampler=self._micgeom_sampler, sampler_id=1)
        else:
            def add_mic_pos(data):
                data["mic_pos"] = self.mic_pos
                return data
            return add_mic_pos

    def _build_signal_seed_map_fn(self):
        self.create_signal_seed_sampler()
        return partial(signal_seed_sampler_fn, sampler=self._signal_seed_sampler, sampler_id=2)

    def _build_rms_map_fn(self):
        self.create_rms_sampler()
        return partial(rms_sampler_fn, sampler=self._rms_sampler, sampler_id=3)

    def _build_loc_map_fn(self):
        self.create_location_sampler()
        return partial(loc_sampler_fn, sampler=self._location_sampler, sampler_id=4)

    def _build_mic_noise_map_fn(self):
        if self.mic_sig_noise:
            self.create_mic_noise_sampler()
            return partial(mic_noise_variance_sampler_fn, sampler=self._mic_noise_sampler, sampler_id=5)
        else:
            def add_mic_noise(data):
                data["mic_noise_variance"] = 0
                return data
            return add_mic_noise

    def _build_signal_length_map_fn(self):
        if self.random_signal_length:
            self.create_signal_length_sampler()
            return partial(signal_length_sampler_fn, sampler=self._signal_length_sampler, sampler_id=6)
        else:
            def add_signal_length(data):
                data["signal_length"] = self.signal_length
                return data
            return add_signal_length

    def build_map_fn(self, split="training"):
        split_fn = partial(add_split, split=split)
        # create stateless sampling functions
        fns = []
        fns.append(self._build_nsources_map_fn())
        fns.append(self._build_mic_pos_map_fn())
        fns.append(self._build_signal_seed_map_fn())
        fns.append(self._build_loc_map_fn())
        fns.append(self._build_rms_map_fn())
        fns.append(self._build_signal_length_map_fn())
        fns.append(self._build_mic_noise_map_fn())

        def map_fn(data): #TODO: restrict this function to use a single thread?
            data = split_fn(data)
            for fn in fns:
                data = fn(data)
            return data
        return map_fn



class MonteCarloFactory(BaseModel):
    """Factory class for creating Monte Carlo implementations."""

    model_type: str = Field(..., description="Type of Monte Carlo model (e.g., 'synthetic').")

    @staticmethod
    def configure_model(**kwargs: Dict[str, Any]) -> MonteCarloBase:
        """Create a specific Monte Carlo model based on the model type."""
        model_type = kwargs.pop("model_type")
        model_class = MONTE_CARLO_FACTORY_MAPPING.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported Monte Carlo model type: {model_type}")
        return model_class(**kwargs)


def register_monte_carlo_model(model_type: str, model_class: Type[MonteCarloBase]):
    """Register a custom Monte Carlo model."""
    MONTE_CARLO_FACTORY_MAPPING[model_type] = model_class



MONTE_CARLO_FACTORY_MAPPING = {
    "synthetic": MonteCarloSynthetic
}

if __name__ == "__main__":

    mc = MonteCarloSynthetic()
    func = mc.build_map_fn(split="training")
    res = func({"idx": 0})
    res1 = func({"idx": 1})
    print(res["rms"], res1["rms"])



