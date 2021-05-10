import unittest
from parameterized import parameterized
import pandas as pd
from scipy.stats import norm
import ray
from numpy import array
from numpy.random import RandomState, default_rng, seed
from numpy.testing import assert_almost_equal
from acoupipe import * 
from pipeline_value_test import get_pipeline, get_distributed_pipeline
from acoular import WNoiseGenerator, PointSource, SourceMixer, MicGeom
from parameterized import parameterized

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
    """method used to test ContainerSampler class"""
    def sample_method(random_state):
        target_instance.attribute = random_state.random()
    return sample_method

FLOAT_SET = [0.1,0.2,0.3,0.4] # a set with floats for testing


class Test_SetSampler(unittest.TestCase):

    def setUp(self):
        self.sampler = SetSampler(random_state=RandomState(1),
                    set = FLOAT_SET,
                    attribute="attribute",
                    replace=False)
        self.sampler.target = [Target() for i in range(3)]

    @parameterized.expand([
        ["single_value_False",False],
        ["single_value_True",True],
    ])
    def test_single_value(self, name, single_value):
        """test single_value capabilities"""
        self.sampler.single_value = single_value
        self.sampler.sample()
        samples = [tr.attribute for tr in self.sampler.target]
        if single_value:
            self.assertEqual(len(set(samples)),1)
        if not single_value:
            self.assertTrue(len(set(samples))>1)

    def test_sampling_linked_attributes(self):
        """test assignment to linked attributes"""
        linkedTarget = LinkedTarget()
        target = Target()
        target.attribute = linkedTarget
        self.sampler.target = [target]
        self.sampler.attribute = "attribute.linked_attribute"
        self.sampler.sample()
        self.assertNotEqual(linkedTarget.linked_attribute,0)


class Test_SourceSetSampler(unittest.TestCase):

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
    def test_source_set_sampling(self,name,single_value,numsamples,expected_setsize):
        self.sampler.random_state = RandomState(1)
        self.sampler.numsamples=numsamples
        self.sampler.single_value=single_value
        self.sampler.sample()
        l1 = len(self.sampler.target[0].sources)
        l2 = len(self.sampler.target[1].sources)
        self.assertEqual((l1,l2),(expected_setsize,expected_setsize))
        if single_value: # assert that same sources at every target
            self.assertTrue(self.sampler.target[0].sources==self.sampler.target[1].sources)
        else:
            self.assertTrue(self.sampler.target[0].sources!=self.sampler.target[1].sources)


class Test_NumericAttributeSampler(Test_SetSampler):

    def setUp(self):
        self.sampler = NumericAttributeSampler(random_var=norm(loc=0,scale=0.1688),random_state=5,
                    attribute="attribute")
        self.sampler.target = [Target() for i in range(10)]

    def test_order(self):
        """ assert that ordering of numeric samples works """
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
        """ verifies that normalization works """
        self.sampler.normalize = True
        self.sampler.sample() 
        l = [t.attribute for t in self.sampler.target]
        self.assertEqual(max(l),1.0)



class Test_ContainerSampler(unittest.TestCase):

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
        """test if BasePipeline can handle ContainerSampler with given random_seeds"""
        pipeline = BasePipeline(
            sampler = [self.containerSampler],
            random_seeds=[range(1,10)],
            features={"random_values": lambda: self.target.attribute})
        data = list(pipeline.get_data())
        for j,d in enumerate(data):
            self.assertEqual(d["random_values"],default_rng(j+1).random())

    def test_pipelining_without_seeds(self):
        """test if BasePipeline can handle ContainerSampler without given random_seeds"""
        rng1 = RandomState(100)
        rng2 = RandomState(100)
        self.containerSampler.random_state = rng1
        pipeline = BasePipeline(
            sampler = [self.containerSampler],
            random_seeds=[],
            features={"random_values": lambda: self.target.attribute})
        data = list(pipeline.get_data())
        for j,d in enumerate(data):
            self.assertEqual(d["random_values"],rng2.random())

    def test_error_handling(self):
        """tests if ValueError is thrown for wrong random_func input"""
        def random_func(rng,x):
            pass
        self.containerSampler.random_func = random_func
        self.assertRaises(ValueError,self.containerSampler.sample)

class Test_MicGeomSampler(unittest.TestCase):

    def get_micgeom(self):
        rng = RandomState(1)
        mics = MicGeom(mpos_tot=rng.rand(3,10))
        return mics

    def get_sampler(self,stype="deviate"):
        rng = RandomState(2) 
        normal_distribution = norm(loc=0, scale= 0.1) 
        if stype == 'deviate':
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                ddir = array([[1.],[1.],[1.]]))
        elif stype == 'rotate':
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                rvec = array([[1.],[1.],[1.]])) 
        elif stype == 'translate':
            sampler = MicGeomSampler(random_var=normal_distribution,
                                random_state=rng,
                                tdir = array([[1.],[1.],[1.]])) 
        return sampler

    def test_mpos_init(self):
        """1. test that mpos_init has not changed after sampling. 
           2. test that mpos changed due to sampling.
           3. test that digest of MicGeom object has changed after sampling. """
        for mode in ['deviate','rotate','translate']:
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


class Test_Sampler(unittest.TestCase):

    def test_instancing(self):
        """create an instance of each class defined in module"""
        for c in SAMPLER_CLASSES:
            c()

    def test_seeding(self):
        """tests if different random states can be assigned to sampler objects"""
        for c in SAMPLER_CLASSES:
            for state in STATES:
                if not (c in [SetSampler,SourceSetSampler,ContainerSampler]):
                    c(random_state=state)



class Test_BasePipeline(unittest.TestCase):

    def setUp(self):
        """will be called for every single test"""
        self.pipeline = get_pipeline(100, mfile="array64_d0o686.xml")
        self.test_seeds = [range(1,1+100),range(2,2+100),range(3,3+100),range(4,4+100)]

    def tearDown(self):
        """will be called after every single test"""
        pass

    def compare_values(self,df,data):
        """compares reference values saved in csv file with samples values one by one"""
        for d in data:
            for key in d.keys(): # "num_sources" etc.
                idx = d["idx"]
                desired = str(df[key][idx-1])
                actual = str(d[key])
                self.assertEqual(actual,desired)

    def test_too_short_random_seeds_input(self):
        """test that exceptions are raised on too short random_seeds input"""
        self.pipeline.random_seeds = self.test_seeds[:1]
        self.assertRaises(ValueError,lambda: next(self.pipeline.get_data()))
    
    def test_non_equal_length_random_seeds_input(self):
        """test that exceptions are raised on random_seeds input of non-equal length"""
        self.test_seeds[0] = range(0,10)
        self.pipeline.random_seeds = self.test_seeds
        self.assertRaises(ValueError,lambda: next(self.pipeline.get_data()))

    def test_sampled_values_without_seeds(self):
        """verifies that sampled values do not change across versions of code without using the seeds argument"""
        df = pd.read_csv("test_data.csv") # reference values
        data = list(self.pipeline.get_data())
        self.assertEqual(len(data),100) # assure correct dataset size TODO: write extra test
        self.compare_values(df,data) # compare all values

    def test_sampled_values_with_seeds(self):
        """verifies that sampled values do not change across versions of code with using the random_seeds argument"""
        self.pipeline.random_seeds = self.test_seeds
        df = pd.read_csv("test_data_seeds.csv") # reference values
        data = list(self.pipeline.get_data())
        self.assertEqual(len(data),100) # assure correct dataset size TODO: write extra test
        self.compare_values(df,data) # compare all values


class Test_DistributedPipeline(Test_BasePipeline):

    def setUp(self):
        """will be called for every single test"""
        ray.init()
        self.pipeline = get_distributed_pipeline(100,2, mfile="array64_d0o686.xml") # two workers
        self.test_seeds = [range(1,1+100),range(2,2+100),range(3,3+100),range(4,4+100)]

    def tearDown(self):
        """will be called after every single test"""
        ray.shutdown()


if __name__ == '__main__':
    unittest.main()
