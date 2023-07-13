"""Random processes to sample values according to a specified random distribution (random variable).

.. autosummary::
    :toctree: generated/

    BaseSampler
    NumericAttributeSampler
    SetSampler
    SourceSetSampler
    ContainerSampler
    LocationSampler
    PointSourceSampler
    MicGeomSampler
    CovSampler
    SpectraSampler

"""

from inspect import signature

from acoular import MicGeom, PointSource, SamplesGenerator, SourceMixer
from numpy import array, cos, diag, empty, eye, pi, repeat, sin, sort, sum, zeros
from numpy.random import Generator, RandomState
from scipy.stats import _distn_infrastructure
from traits.api import (
    Any,
    Bool,
    Callable,
    CArray,
    Either,
    Enum,
    Float,
    HasPrivateTraits,
    Instance,
    Int,
    List,
    Property,
    Str,
    Trait,
    Tuple,
    cached_property,
    on_trait_change,
)

from acoupipe.filter import generate_uniform_parametric_eq


class BaseSampler(HasPrivateTraits):
    """Base class that represents a random process.

    This class has no functionality and should not be used in practice.
    Manipulates attributes of an instance or a list of instances according to a specified random distribution.
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
    single_value = Bool(False,
        desc="manages if a single value is chosen for all targets")

    def rvs(self, size=1):
        """Random variable sampling (for internal use)."""
        return self.random_var.rvs(size=size, random_state=self.random_state)

    def sample(self):
        """Utilizes :meth:`rvs` function to draw random values from :attr:`random_var` (no functionality in this class)."""
        self.rvs()


class NumericAttributeSampler(BaseSampler):
    """Samples attributes of numeric type (e.g. int, float).

    This class samples attributes of numeric type (e.g. int, float) of an instance or a list of instances according
    to a specified random distribution.
    The attribute to be sampled is specified by :attr:`attribute`.
    The sampled values are normalized to the range [0,1] if :attr:`normalize` is set to True.
    The sampled values are ordered in ascending or descending order for all objects in the :attr:`target` list if :attr:`order`
    is set to "ascending" or "descending". If no value is set (:attr:`order` `=None`), no ordering is performed.
    """

    #: attribute of the object in the :attr:`target` list that should be
    #: sampled by the random variable
    attribute = Str(desc="name of the target instance attribute to be manipulated (sampled)")

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
        """Internal function to order drawn values."""
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
        """Random sampling of the target instance attribute.

        Utilizes :meth:`rvs` function to draw random values from :attr:`random_var` that are going to be assigned
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
    """Draws one or multiple values from a given set of values.

    This class draws one or multiple values from a given set of values.
    The set of values is specified by :attr:`set`.
    The number of samples to be drawn is specified by :attr:`numsamples`.
    The attribute to be sampled is specified by :attr:`attribute`.
    A list with sample probabilities can be specified by :attr:`probabilities`.
    The attribute :attr:`replace` specifies if the same object in :attr:`set`
    can be drawn multiple times.
    """

    #: not needed for this sampler; relies on numpy choice function.
    random_var = Enum(None)

    #: the state of the random variable (defaults to :class:`numpy.random.RandomState` )
    random_state = Either(Generator,RandomState,default=RandomState(),
        desc="random state of the random variable")

    #: attribute of the object in the :attr:`target` list that should be
    #: sampled by the random variable
    attribute = Str(desc="name of the target instance attribute to be manipulated (sampled)")

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

    def _get_sample_propabilities(self):
        """Return propabilities associated with the samples in the given set (for internal use)."""
        if not self.prob_list:
            prob_list = None
        else:
            prob_list = self.prob_list
        return prob_list

    def set_value(self, target, value):
        if len(self.attribute.split("."))== 1:
            setattr(target, self.attribute, value[0])
        else:
            asub1 = self.attribute.split(".")[:-1]
            asub2 = self.attribute.split(".")[-1]
            setattr(eval("target."+".".join(asub1)), asub2, value[0])

    def rvs(self, size=1):
        """Random variable sampling (for internal use)."""
        prob_list = self._get_sample_propabilities()
        value = self.random_state.choice(self.set, size=size,
                replace=self.replace, p=prob_list)
        return value

    def sample(self):
        """Random sampling of the target instance attribute.

        Utilizes :meth:`rvs` function to draw random values from :attr:`set` that are going to be assigned
        to the target instance attribute :attr:`attribute` via internal :meth:`set_value` method.
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
    """Draws one or multiple sources of type :class:`acoular.SamplesGenerator` from a given set of sources.

    From a given set of sources (type :class:`acoular.SamplesGenerator`),
    :class:`SourceSetSampler` draws one or multiple sources from this set
    and assigns those to one or more SourceMixer instances. The number of sources to be drawn is specified by
    :attr:`nsources`. The attribute to be sampled is specified by :attr:`attribute`.
    """

    #: a list of :class:`acoular.SourceMixer` instances
    target = List(Instance(SourceMixer, ()),
        desc="the SourceMixer instances holding a subset of sources")

    # a list of :class:`acoular.SamplesGenerator` instances representing the set of sources
    set = List(Instance(SamplesGenerator, ()),
            desc="set of sources to be drawn")

    #: number of samples to be drawn from the set
    nsources = Int(1,
        desc="number of sources to be drawn from the set")

    #: attribute where drawn sources are assigned to. Fixed to :attr:`sources` attribute
    #: of :class:`acoular.SourceMixer`.
    attribute = Enum("sources",
        desc="class instance samples the sources attribute of a SourceMixer instance")

    def sample(self):
        """Random sampling of sources.

        Utilizes :meth:`rvs` function to draw sources from :attr:`set` that are going to be assigned
        to the :class:`acoular.SourceMixer` instance.
        """
        if self.single_value:
            samples = self.rvs(self.nsources)
            for target in self.target:
                target.sources = list(samples)
        else:
            for target in self.target:
                samples = self.rvs(self.nsources)
                target.sources = list(samples)


class ContainerSampler(BaseSampler):
    """Special case of a Sampler to enable the use of an arbitrary sampling function.

    Takes an arbitrary callable with the signature '<Signature (numpy.random.RandomState)>' or
    '<Signature (numpy.random.Generator)>'.
    The callable is evoked via the :meth:`sample` method of this class.
    The output of the callable is assigned to the :attr:`target` attribute.
    """

    target = Any(
        desc="output of the callable is assigned to this attribute when calling the sample method")

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
        """Evokes the :attr:`random_func`."""
        self._validate()
        return self.random_func(self.random_state)

    def sample(self):
        """Random sampling.

        this function utilizes :meth:`rvs` function to evaluate the :attr:`random_func`.
        The output of the :attr:`random_func` can be accessed with the :attr:`target` attribute.
        """
        self.target = self.rvs()


class LocationSampler(BaseSampler):

    #: locations
    target = CArray(
        desc="array with source locations")

    #: the number of source for which the location is sampled
    nsources = Int(
        desc="number of sources")

    #: the random variable specifying the random distribution
    random_var = Tuple(
        desc="instance of a random variable from scipy.stats module")

    #: limits of the allowed locations of a source along the x-axis (lower,upper)
    x_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the x-axis (lower,upper)")

    #: limits of the allowed locations of a source along the y-axis (lower,upper)
    y_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the y-axis (lower,upper)")

    #: limits of the allowed locations of a source along the z-axis (lower,upper)
    z_bounds = Tuple(None, None,
        desc="limits of the allowed drawn locations along the x-axis (lower,upper)")

    def _bounds_violated(self,loc):
        """Validation of drawn source locations."""
        if self.x_bounds[0]:
            if (self.x_bounds[0] > loc[0]): return True
        if self.x_bounds[1]:
            if (loc[0] > self.x_bounds[1]): return True
        if self.y_bounds[0]:
            if (self.y_bounds[0] > loc[1]): return True
        if self.y_bounds[1]:
            if (loc[1] > self.y_bounds[1]): return True
        if self.z_bounds[0]:
            if (self.z_bounds[0] > loc[2]): return True
        if self.z_bounds[1]:
            if (loc[2] > self.z_bounds[1]): return True
        else:
            return False

    def rvs(self):
        """Random variable sampling (for internal use)."""
        return array([
            self.random_var[0].rvs(size=1, random_state=self.random_state),
            self.random_var[1].rvs(size=1, random_state=self.random_state),
            self.random_var[2].rvs(size=1, random_state=self.random_state),
        ]).squeeze()

    def sample(self):
        """Random sampling of locations."""
        loc_array = empty((3,self.nsources))
        for i in range(self.nsources):
            new_loc = self.rvs()
            while self._bounds_violated(new_loc):
                new_loc = self.rvs()
            else:
                loc_array[:,i] = new_loc
        self.target = loc_array



class PointSourceSampler(LocationSampler):
    """Random process that samples the locations of one or more instances of type :class:`PointSource`."""

    #: a list of :class:`acoular.PointSource` instances
    target = Trait(list,
        desc="a list of PointSource instances to manipulate")

    #: the random variable specifying the random distribution
    random_var = Instance(_distn_infrastructure.rv_frozen,
        desc="instance of a random variable from scipy.stats module")

    #:manages if a single value is chosen for all targets
    #: is fixed to False (one value for each object in :attr:`target` list is drawn)
    single_value = Enum(False,
        desc="manages if the same sampled value is assigned to all targets; (only False is valid)")

    #: (x,y,z)-directions of location sampling
    ldir = CArray( dtype=float, shape=(3, (1, 3)),
        desc="(x,y,z)-directions of location sampling")

    @on_trait_change("target")
    def validate_target(self):
        for t in self.target:
            if not isinstance(t,PointSource):
                raise AttributeError("Elements in target must be instances of class acoular.PointSource")

    def sample_loc(self, target):
        """Sampling of a single target location (internal use)."""
        loc_axs = self.ldir.nonzero()[0] # get axes to sample
        loc = array(target.loc)
        loc[loc_axs] = self.ldir[loc_axs].squeeze() * self.rvs(size=loc_axs.size)
        return loc

    def rvs(self, size=1):
        """Random variable sampling (for internal use)."""
        return self.random_var.rvs(size=size, random_state=self.random_state)

    def sample(self):
        """Random sampling of :class:`acoular.PointSource` locations."""
        if self.ldir.any():
            for target in self.target:
                new_loc = self.sample_loc(target)
                while self._bounds_violated(new_loc):
                    new_loc = self.sample_loc(target)
                else:
                    target.loc = tuple(new_loc)



class MicGeomSampler(BaseSampler):
    """Random disturbance of microphone positions of a :class:`acoular.MicGeom` object."""

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

    cpm = Property(
        depends_on=["rvec"],
        desc="cross-product matrix used in Rodrigues' rotation formula")

    @cached_property
    def _get_mpos_init(self):
        return self.target.mpos_tot.copy()

    @cached_property
    def _get_cpm(self):
        [[x], [y], [z]] = self.rvec
        return array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def rotate(self):
        """Rotates the microphone array."""
        theta = 2 * pi * self.rscale * self.rvs()
        # Rodrigues' rotation formula
        R = eye(3) + sin(theta) * self.cpm + (1 - cos(theta)) * self.cpm @ self.cpm
        new_mpos_tot = R @ self.target.mpos_tot.copy()
        self.target.mpos_tot = new_mpos_tot.copy()

    def translate(self):
        """Translates the microphone array."""
        new_mpos_tot = self.target.mpos_tot.copy()
        new_mpos_tot += sum(self.tdir *self.rvs(size=self.tdir.shape[-1]),
            axis=-1).reshape(-1, 1)
        self.target.mpos_tot = new_mpos_tot.copy()

    def deviate(self):
        """Deviates the individual microphone positions."""
        dev_axs = self.ddir.nonzero()[0] # get axes to deviate
        new_mpos_tot = self.target.mpos_tot.copy()
        new_mpos_tot[dev_axs] += self.ddir[dev_axs] * \
            self.rvs(size=((self.mpos_init.shape[1],dev_axs.size))).T
        self.target.mpos_tot = new_mpos_tot.copy()

    def sample(self):
        """Random sampling of microphone positions."""
        self.target.mpos_tot = self.mpos_init.copy() # initialize
        if self.rvec.any():  # if rotation vector exist, rotate first!
            self.rotate()
        if self.tdir.any():
            self.translate()
        if self.ddir.any():
            self.deviate()


class CovSampler(BaseSampler):
    """Sampler to sample covariance matrices for a specific number of sources.

    The current implementation only allows uncorrelated sources, meaning that the
    sampled covariances matrices at :attr:`target` are diagonal matrices.
    The strength (variance) of the sources follows the given random distribution at :attr:`random_var`.
    The attribute :attr:`nsources` determines the number of sources to be sampled.
    The :attr:`nfft` attribute determines the number of fft bins at which the power is distributed.
    The power of each source is sampled from the given random distribution at :attr:`random_var`.
    and assigned to :attr:`variances` after sampling. The attribute :attr:`single_value` determines if a single
    power is chosen for all sources.
    The :attr:`scale_variance` attribute determines if the variance is scaled such that the sum of variances equals to 1.
    """

    #: the sampled complex covariance matrices of shape (nfft, nsources, nsources)
    target = CArray(
        desc="the sampled complex covariance matrices")

    #: the sampled variances of shape (nsources,)
    variances = CArray(
        desc="the sampled variances"
        )

    #: the number of sources to be sampled
    nsources = Int(
        desc="the number of sources to be sampled")

    #: the number of fft bins at which the power is distributed
    nfft = Int(1,
        desc="number of fft bins at which the power is distributed")

    #: True: same amplitudes for all sources
    single_value = Bool(False,
        desc="manages if a single amplitude is chosen for all sources")

    #: True: sum of variances is 1
    scale_variance = Bool(False)

    def sample(self):
        """Random sampling of covariance matrices.

        Utilizes :meth:`rvs` function to draw random values from :attr:`random_var`.
        Output of the :meth:`rvs` function is the covariance matrix of the sources
        and is assigned to :attr:`target`.
        """
        if self.single_value:
            variance = self.rvs(size=1)
            if self.scale_variance: # sum of variances is 1
                variance = 1/self.nsources
            variance = repeat(variance, self.nsources)
        else:
            variance = self.rvs(size=self.nsources)
            if self.scale_variance: # sum of variances is 1
                variance /= variance.sum()
        self.variances = variance.copy() # copy full variance
        variance /= self.nfft
        variance = diag(variance.astype(complex))
        self.target = repeat(variance,self.nfft).reshape((self.nsources,self.nsources,self.nfft)).T


class SpectraSampler(CovSampler):
    """Random sampling of power spectra.

    The current implementation only allows uncorrelated sources, meaning that the
    sampled power spectra at :attr:`target` are diagonal matrices.
    The attribute :attr:`nsources` determines the number of sources to be sampled.
    The :attr:`nfft` attribute determines the number of fft bins at which the power is distributed.
    The power of each source is sampled from the given random distribution at :attr:`random_var`.
    and assigned to :attr:`variances`. The attribute :attr:`single_value` determines if a single
    power is chosen for all sources. The :attr:`scale_variance` attribute determines if the
    variance is scaled such that the sum of variances equals to 1.
    The attribute :attr:`single_spectra` determines if the same underlying transfer function of
    a power spectrum is assigned to all sources. The attribute :attr:`max_order` determines the
    maximum order of the power spectra filter. The sampled Power Spectra are assigned to :attr:`target`
    after sampling.
    """

    #: the sampled power spectra of shape (nfft, nsources, nsources)
    target = CArray(dtype=complex,
        desc="the sampled auto and cross-power spectra")

    #: True: same spectra for all sources
    single_spectra = Bool(False,
        desc="individual spectra for each source")

    #: the maximum order of the power spectra filter
    max_order = Int(16,
        desc="maximum order of the power spectra filter")

    _random_state = Either(RandomState, Generator)

    @on_trait_change("random_state")
    def copy_random_state(self):
        state = self.random_state.get_state()
        self._random_state = RandomState()
        self._random_state.set_state(state)

    def sample(self):
        """Utilizes :meth:`rvs` function to evaluate the :attr:`random_func`."""
        if self.single_value:
            variance = self.rvs(size=1)
            if self.scale_variance: # sum of variances is 1
                variance = 1/self.nsources
            variance = repeat(variance, self.nsources)
        else:
            variance = self.rvs(size=self.nsources)
            if self.scale_variance: # sum of variances is 1
                variance /= variance.sum()
        Q = zeros((self.nfft,self.nsources,self.nsources),dtype=complex)
        if self.single_spectra:
            Hw,_ = generate_uniform_parametric_eq(self.nfft, self.max_order, self._random_state)
            Hw = Hw.conj()/(Hw.conj() * Hw) # invert filter response
            Hw2 = Hw*Hw.conj()
            Hw2 /= Hw2.sum()
            for i in range(self.nsources):
                Q[:,i,i] = Hw2*variance[i].astype(complex)
        else:
            for i in range(self.nsources):
                Hw, _ = generate_uniform_parametric_eq(self.nfft, self.max_order, self._random_state)
                Hw = Hw.conj()/(Hw.conj() * Hw) # invert filter response
                Q[:,i,i] = Hw*Hw.conj()
                Q[:,i,i] /= Q[:,i,i].sum()/variance[i].astype(complex)
        self.target = Q
        self.variances = variance.copy() # copy full variance


