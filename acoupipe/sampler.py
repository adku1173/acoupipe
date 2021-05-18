# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2020-2023, Adam Kujawski, Simon Jekosch, Art Pelling, Ennes Sarradj.
#------------------------------------------------------------------------------
"""
All classes in this module are random processes that meant to be used to manipulate instances
i.e. their attribute values according to a specified random distribution (random variable). 

.. autosummary::
    :toctree: generated/

    BaseSampler
    NumericAttributeSampler
    SetSampler
    SourceSetSampler
    ContainerSampler
    PointSourceSampler
    MicGeomSampler

"""

from traits.api import HasPrivateTraits, Instance, CArray, Float, Property,\
    Str, cached_property, Any, Tuple, List, Bool, Enum, Trait, Int, Either, Type,Callable,\
    on_trait_change
from acoular import MicGeom, PointSource, SourceMixer, SamplesGenerator
from numpy import array, pi, sin, cos, sum, eye, sort
from numpy.random import default_rng, RandomState, Generator
from scipy.stats import _distn_infrastructure
from inspect import signature


class BaseSampler(HasPrivateTraits):
    """Base class that represents a random process manipulating attributes of an instance or a list of
    instances according to a specified random distribution.

    This class has no functionality and should not be used in practice.
    """

    #: a list of instances which attributes are to be manipulated
    target = Trait(list,
        desc="the object instances which are manipulated by this class")

    #: the random variable specifying the random distribution
    random_var = Instance(_distn_infrastructure.rv_frozen,
        desc="instance of a random variable from scipy.stats module")

    #: the state of the random variable :attr:`random_var`
    random_state = Either(int,RandomState,Generator,
        desc="random state of the random variable")

    #: manages if the same sampled value is chosen for all objects in the :attr:`target` list 
    #: if False, one value for each target is drawn
    single_value = Bool(False, #TODO: better rename to same_sample
        desc="manages if a single value is chosen for all targets")

    def rvs(self, size=1):
        """random variable sampling (for internal use)"""
        return self.random_var.rvs(size=size, random_state=self.random_state)

    def sample(self):
        """this function utilizes :meth:`rvs` function to draw random values
        from :attr:`random_var` (no functionality in this class). 
        """
        self.rvs()

  
class NumericAttributeSampler(BaseSampler):
    """Random process that manipulates attributes of numeric type (e.g. int, float) 
    according to a specified random distribution.
    """
    
    #: attribute of the object in the :attr:`target` list that should be 
    #: sampled by the random variable
    attribute = Str("", 
        desc="name of the target instance attribute to be manipulated (sampled)")

    #: whether to normalize the drawn values (maximum element equals 1).
    #: if :attr:`single_value` is set to True, this has no effect.
    normalize = Bool(False, 
        desc="if attribute is True, sampled values will be normalized")
    
    #: whether to order the drawn values in ascending or descending order for all objects in the :attr:`target` list.
    #: if :attr:`single_value` is set to True, this has no effect. If no value is set (:attr:`order` `=None`), no ordering is performed.
    order = Either("ascending","descending")

    #: sampled value filter (resample if callable filter returns False)
    filter = Callable(
        desc="a callable function that returns a bool"
        )

    def order_samples(self, samples):
        """internal function to order drawn values"""
        samples = sort(samples)
        if self.order == "descending":
            samples = samples[::-1]
        elif self.order == "ascending":
            pass
        return samples

    def set_value(self, target, value):
        if len(self.attribute.split("."))== 1:
            setattr(target, self.attribute, value)
        else:
            asub1 = self.attribute.split(".")[:-1]
            asub2 = self.attribute.split(".")[-1]
            setattr(eval("target."+".".join(asub1)), asub2, value)

    def sample(self):
        """this function utilizes :meth:`rvs` function to draw random 
        values from :attr:`random_var` that are going to be assigned 
        to the target instance attribute via internal :meth:`set_value` method.
        """
        if self.single_value:
            value = self.rvs()[0] 
            if self.filter: # resample if filter returns False
                while not self.filter(value):
                    value = self.rvs()[0] 
            for target in self.target:       
                self.set_value(target, value)   
        else:
            values = self.rvs(len(self.target))
            if self.filter: # resample if filter returns False
                while not self.filter(values):
                    values = self.rvs(len(self.target))
            if self.normalize:
                values /= max(abs(values))
            if self.order:
                values = self.order_samples(values)
            for i, target in enumerate(self.target):
                self.set_value(target, values[i])


        
class SetSampler(BaseSampler):
    """
    Random process that draws one or multiple values 
    from a given set of values and assigns those to the target. 
    """
    
    #: not needed for this sampler; relies on numpy choice function.
    random_var = Enum(None)

    #: the state of the random variable (defaults to :class:`numpy.random.RandomState` )
    random_state = Either(Generator,RandomState,default=RandomState(),
        desc="random state of the random variable")
                        
    #: attribute of the object in the :attr:`target` list that should be 
    #: sampled by the random variable
    attribute = Str("", 
        desc="name of the target instance attribute to be manipulated (sampled)")
    
    #: a List of Samples representing the set
    set = List([], 
        desc="set of samples to be drawn")
    
    #: number of samples to be drawn from the set (fixed to one)
    numsamples = Enum(1,
        desc="number of samples to be drawn from the set (behavior is fixed to a single sample)")

    #: whether to use the replace argument of :meth:`numpy.random.choice` function
    replace = Bool(True, 
        desc="replace option for Choice function")

    #:Probability List, same lenght as the set, need to sum up to 1
    #:If left empty assume uniform distribution 
    prob_list =  List([], 
        desc="Probability List, same lenght as the set, need to sum up to 1")

    def get_sample_propabilities(self):
        """return propabilities associated with the samples in the 
        given set (for internal use)."""
        if not self.prob_list:
            prob_list = None 
        else:
            prob_list = self.problist  
        return prob_list        

    def set_value(self, target, value):
        if len(self.attribute.split("."))== 1:
            setattr(target, self.attribute, value[0])
        else:
            asub1 = self.attribute.split(".")[:-1]
            asub2 = self.attribute.split(".")[-1]
            setattr(eval("target."+".".join(asub1)), asub2, value[0])

    def rvs(self, size=1):
        """random variable sampling (for internal use)"""
        prob_list = self.get_sample_propabilities()
        value = self.random_state.choice(self.set, size=size,
                replace=self.replace, p=prob_list)
        return value

    def sample(self):
        """this function utilizes :meth:`rvs` function to draw  
        values from :attr:`set` list that are going to be assigned 
        to the target instance attribute.
        """
        # draw a single value from set -> assign to each target in target List
        if self.single_value:
            samples = self.rvs(self.numsamples)
            for target in self.target:       
                self.set_value(target, samples)
        else:
            for target in self.target:
                samples = self.rvs(self.numsamples)
                self.set_value(target, samples)


class SourceSetSampler(SetSampler):
    """
    From a given set of sources (type :class:`acoular.SamplesGenerator`), 
    :class:`SourceSetSampler` draws one or multiple sources from this set
    and assigns those to one or more SourceMixer instances.
    """
    
    #: a list of :class:`acoular.SourceMixer` instances 
    target = List(Instance(SourceMixer, ()),
        desc="the SourceMixer instances holding a subset of sources")
    
    # a list of :class:`acoular.SamplesGenerator` instances representing the set of sources
    set = List(Instance(SamplesGenerator, ()),
            desc="set of sources to be drawn") 

    #: number of samples to be drawn from the set
    numsamples = Int(1,
        desc="number of samples to be drawn from the set")

    #: attribute where drawn sources are assigned to. Fixed to :attr:`sources` attribute
    #: of :class:`acoular.SourceMixer`.
    attribute = Enum("sources", 
        desc="class instance samples the sources attribute of a SourceMixer instance")    

    def set_value(self, target, value):
        """assigns the sampled values to the SourceMixer instances; only for internal use"""
        #target.sources=[s for s in value] # convert numpy array holding objects into list
        [target.sources.append(s) for s in value]

    def sample(self):
        """this function utilizes :meth:`rvs` function to draw  
        values from :attr:`set` list that are going to be assigned 
        to the target instance attribute.
        """
        # draw a single value from set -> assign to each target in target List
        if self.single_value:
            samples = self.rvs(self.numsamples)
            for target in self.target:      
                target.sources.clear() # remove sources 
                self.set_value(target, samples)
        else:
            for target in self.target:
                target.sources.clear() # remove sources 
                samples = self.rvs(self.numsamples)
                self.set_value(target, samples)


class ContainerSampler(BaseSampler):
    """Special case of a Sampler to enable the use 
    of an arbitrary function performing the sampling and that can be passed to a
    :class:`BasePipeline` derived class.
    
    This class has no explicit target. Instead it takes an arbitrary callable 
    with the signature '<Signature (numpy.random.RandomState)>' or 
    '<Signature (numpy.random.Generator)>'. 
    The callable is evoked via the :meth:`sample` method of this class. 
    """

    target = Enum(None,
        desc="this class has no explicit target")

    random_var = Enum(None,
        desc="this class has no explicit random variable")

    #: a callable function that defines the random process.
    #: has to have the signature '<Signature (numpy.random.RandomState)>'
    random_func = Callable(
        desc="the function that defines the random process")

    #: the random state consumed by the :meth:`random_func` callable"
    random_state = Either(RandomState, Generator,
        desc="random state consumed by the func callable")

    def _validate(self):
        if self.random_func:
            sig = signature(self.random_func)
            if len(sig.parameters) == 0 or len(sig.parameters) > 1:
                raise ValueError(
                    "the random_func callable has to have a signature of "
                    "'<Signature (RandomState)>'. Only a single argument "
                    "is valid that takes a random number generator as the "
                    "input.")

    def rvs(self):
        """evokes the :attr:`random_func`"""
        self._validate()
        self.random_func(self.random_state)

    def sample(self):
        """this function utilizes :meth:`rvs` function to evaluate the :attr:`random_func`"""
        self.rvs()

        

class PointSourceSampler(BaseSampler):
    """Random process that samples the locations of one or more 
    instances of type :class:`PointSource`. 
    """
    
    #: a list of :class:`acoular.PointSource` instances 
    target = Trait(list,
        desc="a list of PointSource instances to manipulate")
    
    #:manages if a single value is chosen for all targets 
    #: is fixed to False (one value for each object in :attr:`target` list is drawn)
    single_value = Enum(False,
        desc="manages if the same sampled value is assigned to all targets; (only False is valid)")

    #: limits of the allowed locations of a source along the x-axis (lower,upper)
    x_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the x-axis (lower,upper)") 

    #: limits of the allowed locations of a source along the y-axis (lower,upper)
    y_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the y-axis (lower,upper)") 

    #: limits of the allowed locations of a source along the z-axis (lower,upper)
    z_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the x-axis (lower,upper)") 
    
    #: (x,y,z)-directions of location sampling
    ldir = CArray( dtype=float, shape=(3, (1, 3)), 
        desc="(x,y,z)-directions of location sampling")
    
    def _bounds_violated(self,loc):
        """validation of drawn source locations. 
        Returns False if location exceeds bounds
        """
        if self.x_bounds[0]: 
            if (self.x_bounds[0] >= loc[0]): return True
        if self.x_bounds[1]: 
            if (loc[0] >= self.x_bounds[1]): return True
        if self.y_bounds[0]: 
            if (self.y_bounds[0] >= loc[1]): return True
        if self.y_bounds[1]: 
            if (loc[1] >= self.y_bounds[1]): return True
        if self.z_bounds[0]: 
            if (self.z_bounds[0] >= loc[2]): return True
        if self.z_bounds[1]: 
            if (loc[2] >= self.z_bounds[1]): return True
        else:
            return False

    @on_trait_change("target")        
    def validate_target(self):
        for t in self.target:
            if not isinstance(t,PointSource):
                raise AttributeError("Elements in target must be instances of class acoular.PointSource")

    def sample_loc(self, target):
        """sampling of a single target location (internal use)"""
        loc_axs = self.ldir.nonzero()[0] # get axes to sample
        loc = array(target.loc)
        loc[loc_axs] = self.ldir[loc_axs].squeeze() * self.rvs(size=loc_axs.size)
        return loc
        
    def sample(self):
        """this function utilizes :meth:`rvs` function to draw random 
        values from :attr:`random_var` that are going to be assigned 
        to the :attr:`loc` attribute of a :class:`PointSource` instance.
        """
        if self.ldir.any(): 
            for target in self.target:
                new_loc = self.sample_loc(target)
                while self._bounds_violated(new_loc):
                    new_loc = self.sample_loc(target)
                else:
                    target.loc = tuple(new_loc)
            


class MicGeomSampler(BaseSampler):
    """Random process that samples the microphone positions of one
    instance of type :class:`acoular.MicGeom`.
    """

    #: the microphone geometry instance (type :class:`acoular.MicGeom`)
    target = Instance(MicGeom, 
        desc="microphone geometry whose positions are sampled")

    #:manages if a single value is chosen for all targets 
    #:if False one value for each target is drawn
    single_value = Enum(True, 
        desc="manages if a single value is chosen for all targets")

    #: a copy of the initial microphone geometry object provided as the 
    #: :attr:`target` attribute value.
    mpos_init = Property(depends_on=["target"],
        desc="a copy of the initial microphone geometry")

    #: (x,y,z)-directions of full geometry translation
    tdir = CArray( dtype=float, shape=(3, (1, 3)), 
        desc="(x,y,z)-directions of full geometry translation")

    #: (x,y,z)-directions of individual position deviation
    ddir = CArray(shape=(3, (1, 3)), 
          desc="(x,y,z)-directions of individual position deviation")

    #: Reference vector for rotation
    rvec = CArray( dtype=float, shape=(3,(1, 3)), 
        desc="rotational reference vector")
    
    #: scaling factor of rotation deviation
    rscale = Float(1.0, 
       desc="scaling factor of rotation deviation")

    K = Property(
        depends_on=["rvec"],
        desc="cross-product matrix used in Rodrigues' rotation formula")

    @cached_property
    def _get_mpos_init(self):
        return self.target.mpos_tot.copy()

    @cached_property
    def _get_K(self):
        [[x], [y], [z]] = self.rvec
        return array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def rotate(self):
        """rotates the microphone array"""
        theta = 2 * pi * self.rscale * self.rvs()
        # Rodrigues' rotation formula
        R = eye(3) + sin(theta) * self.K + (1 - cos(theta)) * self.K @ self.K
        new_mpos_tot = R @ self.target.mpos_tot.copy()
        self.target.mpos_tot = new_mpos_tot.copy()

    def translate(self):
        """translates the microphone array"""
        new_mpos_tot = self.target.mpos_tot.copy()
        new_mpos_tot += sum(self.tdir *self.rvs(size=self.tdir.shape[-1]),
            axis=-1).reshape(-1, 1)
        self.target.mpos_tot = new_mpos_tot.copy()

    def deviate(self):
        """deviates the individual microphone positions"""
        dev_axs = self.ddir.nonzero()[0] # get axes to deviate
        new_mpos_tot = self.target.mpos_tot.copy()
        new_mpos_tot[dev_axs] += self.ddir[dev_axs] * \
            self.rvs(size=((self.mpos_init.shape[1],dev_axs.size))).T
        self.target.mpos_tot = new_mpos_tot.copy()

    def sample(self):
        """this function utilizes :meth:`rvs` function to draw random 
        values from :attr:`random_var` that are going to be used to variate 
        the microphone positions of a :class:`MicGeom` instance."""
        self.target.mpos_tot = self.mpos_init.copy() # initialize
        if self.rvec.any():  # if rotation vector exist, rotate first!
            self.rotate()
        if self.tdir.any(): 
            self.translate()
        if self.ddir.any():
            self.deviate()


    
