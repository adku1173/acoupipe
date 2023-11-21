import unittest

from acoular import MicGeom, PointSource, SourceMixer, WNoiseGenerator
from numpy import array
from numpy.random import RandomState, default_rng
from numpy.testing import assert_almost_equal
from parameterized import parameterized
from pipeline_value_test import get_distributed_pipeline, get_pipeline
from scipy.stats import norm

from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.sampler import (
    BaseSampler,
    ContainerSampler,
    MicGeomSampler,
    NumericAttributeSampler,
    PointSourceSampler,
    SetSampler,
    SourceSetSampler,
)

SAMPLER_CLASSES = [BaseSampler, NumericAttributeSampler,
           SetSampler, SourceSetSampler, PointSourceSampler,ContainerSampler, MicGeomSampler]
PIPELINE_CLASSES = [BasePipeline, DistributedPipeline]
STATES = [1, RandomState(1), default_rng(1)]

class LinkedTarget:
    """class that is used in Sampler tests."""

    linked_attribute = 0
class Target:
    """class that is used in Sampler tests."""

    attribute = 0

def create_test_method(target_instance):
    """Method used to test ContainerSampler class."""
    def sample_method(random_state):
        target_instance.attribute = random_state.random()
    return sample_method

FLOAT_SET = [0.1,0.2,0.3,0.4] # a set with floats for testing


class TestSetSampler(unittest.TestCase):

    def setUp(self):
        self.sampler = SetSampler(random_state=RandomState(1),
                    set = FLOAT_SET,
                    attribute="attribute",
                    replace=False)
        self.sampler.target = [Target() for i in range(3)]

    @parameterized.expand([
        ["equal_value_False",False],
        ["equal_value_True",True],
    ])
    def test_equal_value(self, name, equal_value):
        """Test equal_value capabilities."""
        self.sampler.equal_value = equal_value
        self.sampler.sample()
        samples = [tr.attribute for tr in self.sampler.target]
        if equal_value:
            self.assertEqual(len(set(samples)),1)
        if not equal_value:
            self.assertTrue(len(set(samples))>1)

    def test_sampling_linked_attributes(self):
        """Test assignment to linked attributes."""
        linkedTarget = LinkedTarget()
        target = Target()
        target.attribute = linkedTarget
        self.sampler.target = [target]
        self.sampler.attribute = "attribute.linked_attribute"
        self.sampler.sample()
        self.assertNotEqual(linkedTarget.linked_attribute,0)


class TestSourceSetSampler(unittest.TestCase):

    def setUp(self):
        ps_set = [PointSource(signal=WNoiseGenerator(sample_freq=10),mics=MicGeom()) for i in range(4)]
        self.sampler = SourceSetSampler(target=[SourceMixer(),SourceMixer()],
                        set=ps_set,
                        replace=False)

    @parameterized.expand([
        ["case1",False,1,1],
        ["case2",True,1,1],
        ["case3",False,2,2],
        ["case4",False,2,2],
    ])
    def test_source_set_sampling(self,name,equal_value,nsources,expected_setsize):
        self.sampler.random_state = RandomState(1)
        self.sampler.nsources=nsources
        self.sampler.equal_value=equal_value
        self.sampler.sample()
        l1 = len(self.sampler.target[0].sources)
        l2 = len(self.sampler.target[1].sources)
        self.assertEqual((l1,l2),(expected_setsize,expected_setsize))
        if equal_value: # assert that same sources at every target
            self.assertTrue(self.sampler.target[0].sources==self.sampler.target[1].sources)
        else:
            self.assertTrue(self.sampler.target[0].sources!=self.sampler.target[1].sources)


class TestNumericAttributeSampler(TestSetSampler):

    def setUp(self):
        self.sampler = NumericAttributeSampler(random_var=norm(loc=0,scale=0.1688),random_state=5,
                    attribute="attribute")
        self.sampler.target = [Target() for i in range(10)]

    def test_order(self):
        """Assert that ordering of numeric samples works."""
        # proof random order is default
        self.sampler.sample()
        l = [t.attribute for t in self.sampler.target]
        self.assertNotEqual(l[0],min(l),max(l))
        self.assertNotEqual(l[-1],min(l),max(l))
        # proof ascending ordering
        self.sampler.order = "ascending"
        self.sampler.sample()
        l = [t.attribute for t in self.sampler.target]
        self.assertEqual(l[0],min(l))
        self.assertEqual(l[-1],max(l))
        # proof descending ordering
        self.sampler.order = "descending"
        self.sampler.sample()
        l = [t.attribute for t in self.sampler.target]
        self.assertEqual(l[0],max(l))
        self.assertEqual(l[-1],min(l))

    def test_normalization(self):
        """Verifies that normalization works."""
        self.sampler.normalize = True
        self.sampler.sample()
        l = [t.attribute for t in self.sampler.target]
        self.assertEqual(max(l),1.0)



class TestContainerSampler(unittest.TestCase):

    def setUp(self):
        self.target = Target()
        self.containerSampler = ContainerSampler()
        self.rfunc = create_test_method(self.target) # manipulates the target
        self.containerSampler.random_func = self.rfunc

    def test_sampling(self):
        for rgens in [(RandomState(10),RandomState(10)),(default_rng(5),default_rng(5))]:
            (rng1,rng2) = rgens
            self.containerSampler.random_state = rng1
            self.containerSampler.sample() # should change the target attribute
            self.assertEqual(self.target.attribute,rng2.random())

    def test_pipelining_with_seeds(self):
        """Test if BasePipeline can handle ContainerSampler with given random_seeds."""
        pipeline = BasePipeline(
            sampler = {1:self.containerSampler},
            random_seeds={1:range(1,10)},
            features=lambda sm: {"random_values": self.target.attribute})
        data = list(pipeline.get_data(progress_bar=False))
        for j,d in enumerate(data):
            self.assertEqual(d["random_values"],default_rng(j+1).random())

    def test_pipelining_without_seeds(self):
        """Test if BasePipeline can handle ContainerSampler without given random_seeds."""
        rng1 = RandomState(100)
        rng2 = RandomState(100)
        self.containerSampler.random_state = rng1
        pipeline = BasePipeline(
            sampler = {1:self.containerSampler},
            #random_seeds={},
            features=lambda sm: {"random_values": self.target.attribute})
        data = list(pipeline.get_data(progress_bar=False))
        for _j,d in enumerate(data):
            self.assertEqual(d["random_values"],rng2.random())

    def test_error_handling(self):
        """Tests if ValueError is thrown for wrong random_func input."""
        def random_func(rng,x):
            pass
        self.containerSampler.random_func = random_func
        self.assertRaises(ValueError,self.containerSampler.sample)

class TestMicGeomSampler(unittest.TestCase):

    def get_micgeom(self):
        rng = RandomState(1)
        mics = MicGeom(mpos_tot=rng.rand(3,10))
        return mics

    def get_sampler(self,stype="deviate"):
        rng = RandomState(2)
        normal_distribution = norm(loc=0, scale= 0.1)
        if stype == "deviate":
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                ddir = array([[1.],[1.],[1.]]))
        elif stype == "rotate":
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                rvec = array([[1.],[1.],[1.]]))
        elif stype == "translate":
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                tdir = array([[1.],[1.],[1.]]))
        return sampler

    def test_mpos_init(self):
        """Mpos init should be the same as the target mpos_tot.

        1. test that mpos_init has not changed after sampling.
        2. test that mpos changed due to sampling.
        3. test that digest of MicGeom object has changed after sampling.
        """
        for mode in ["deviate","rotate","translate"]:
            with self.subTest(mode):
                micgeom = self.get_micgeom()
                digest1 = micgeom.digest
                sampler = self.get_sampler(mode)
                sampler.target = micgeom
                mpos_target = micgeom.mpos_tot.copy()
                sampler.sample()
                digest2 = micgeom.digest
                assert_almost_equal(mpos_target,sampler.mpos_init)
                self.assertNotEqual(micgeom.mpos_tot[0,0],mpos_target[0,0])
                self.assertNotEqual(digest1,digest2)


class TestSampler(unittest.TestCase):

    def test_instancing(self):
        """Create an instance of each class defined in module."""
        for c in SAMPLER_CLASSES:
            c()

    def test_seeding(self):
        """Tests if different random states can be assigned to sampler objects."""
        for c in SAMPLER_CLASSES:
            for state in STATES:
                if c not in [SetSampler, SourceSetSampler, ContainerSampler]:
                    c(random_state=state)



class TestBasePipeline(unittest.TestCase):

    def setUp(self):
        """Will be called for every single test."""
        self.size = 1
        self.pipeline = get_pipeline(self.size)
        self.test_seeds = {
            1:range(1,1+self.size), 2:range(2,2+self.size), 3:range(3,3+self.size), 4:range(4,4+self.size)}

    def test_pipeline_without_explicit_seeds(self):
        """Test if BasePipeline can handle samplers without given random_seeds."""
        data = next(self.pipeline.get_data(progress_bar=False))
        self.assertTrue(data["data"])

    def test_too_short_random_seeds_input(self):
        """Test that exceptions are raised on too short random_seeds input."""
        seeds = {
            1:range(1,1+self.size)}
        self.pipeline.random_seeds = seeds
        self.assertRaises(ValueError,lambda: next(self.pipeline.get_data(progress_bar=False)))

    def test_non_equal_length_random_seeds_input(self):
        """Test that exceptions are raised on random_seeds input of non-equal length."""
        self.test_seeds[0] = range(0,10)
        self.pipeline.random_seeds = self.test_seeds
        self.assertRaises(ValueError,lambda: next(self.pipeline.get_data(progress_bar=False)))


class TestDistributedPipeline(TestBasePipeline):

    def setUp(self):
        """Will be called for every single test."""
        #ray.shutdown()
        #ray.init(log_to_driver=False)
        self.size = 3
        self.pipeline = get_distributed_pipeline(self.size,2) # two workers
        self.test_seeds = {
            1:range(1,1+self.size), 2:range(2,2+self.size), 3:range(3,3+self.size), 4:range(4,4+self.size)}

    def tearDown(self):
        """Will be called after every single test."""
        #ray.shutdown()


if __name__ == "__main__":
    unittest.main()
