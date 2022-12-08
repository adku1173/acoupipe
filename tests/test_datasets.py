import unittest
from os import path, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # change tensorflow log level for doc purposes
from parameterized import parameterized
from acoupipe.dataset1 import Dataset1
from acoupipe.dataset2 import Dataset2
import numpy as np
import h5py
import tempfile
import shutil
from acoular import MicGeom

dirpath = path.dirname(path.abspath(__file__))

mpos_tot = np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])

class TestDataset1(unittest.TestCase):
   
    dataset_cls = Dataset1

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset = self.dataset_cls(split="training",
                            size=1,
                            f=1000,
                            startsample=20,
                            features=["csm"],
                            mics=MicGeom(mpos_tot=mpos_tot))      
        self.dataset.grid.increment = 1/5*1.5                              
        self.cls_name = type(self.dataset).__name__

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @parameterized.expand([
        ["sourcemap"],
        ["csmtriu"],
        ["csm"],
    ])   
    def test_generate(self, feature):
        self.dataset.features=[feature]
        data = next(self.dataset.generate())
        test_loc = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_loc.npy"))
        test_p2 =  np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_p2.npy"))
        test_feature = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_{feature}.npy"))
        np.testing.assert_allclose(test_loc,data['loc'])
        np.testing.assert_allclose(test_p2,data['p2'])
        np.testing.assert_allclose(test_feature,data[feature])


    @parameterized.expand([
        ["sourcemap"],
        ["csmtriu"],
        ["csm"],
    ])   
    def test_save_h5(self, feature):
        self.dataset.features=[feature]
        self.dataset.save_h5(path.join(self.test_dir,"test.h5"))
        with h5py.File(path.join(self.test_dir,"test.h5"),"r") as file:
            data = file[f"1/{feature}"][:]
            test_feature = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_{feature}.npy"))
            np.testing.assert_allclose(test_feature,data)

    @parameterized.expand([
        ["sourcemap", 1000, 0],
        ["sourcemap", 1000, 3],
        ["sourcemap", [1000, 2000], 3],
        ["sourcemap", [1000, 2000], 0],
        ["sourcemap", None, 0],
        ["sourcemap", None, 3],
        # csmtriu
        ["csmtriu", 1000, 0],
        ["csmtriu", 1000, 3],
        ["csmtriu", [1000, 2000], 3],
        ["csmtriu", [1000, 2000], 0],
        ["csmtriu", None, 0],
        ["csmtriu", None, 3],        
        # csm
        ["csm", 1000, 0],
        ["csm", 1000, 3],
        ["csm", [1000, 2000], 3],
        ["csm", [1000, 2000], 0],
        ["csm", None, 0],
        ["csm", None, 3],               
    ])  
    def test_shapes(self, feature, f, num):
        self.dataset.features=[feature]
        self.dataset.num = num
        self.dataset.f = f
        data = next(self.dataset.generate())
        ndim = 4 # number of sources at this sample
        if f == None:
            fdim = self.dataset.freq_data.fftfreq().shape[0]
        elif type(f) == list:
            fdim = len(f)
        else:
            fdim = 1
        if self.dataset_cls.__name__ == "Dataset1":
            self.assertTupleEqual(data['p2'].shape, (fdim,ndim))
        elif self.dataset_cls.__name__ == "Dataset2":
            self.assertTupleEqual(data['p2'].shape, (fdim,ndim,ndim,2)) # diagonal matrix
        self.assertTupleEqual(data['loc'].shape, (3,ndim))
        if feature == 'csm':
            self.assertTupleEqual(data[feature].shape, (fdim,4,4,2))
        elif feature == 'csmtriu':
            self.assertTupleEqual(data[feature].shape, (fdim,4,4,1))
        else:
            self.assertTupleEqual(data[feature].shape,(fdim,) + self.dataset.grid.shape)       

    def test_nsources(self):
        dataset = self.dataset_cls(split="training",
                                    max_nsources=1,
                                    size=1,
                                    f=1000,
                                    features=["csm","sourcemap","csmtriu"],
                                    mics=MicGeom(mpos_tot=mpos_tot))  
        data = next(dataset.generate())
        if self.dataset_cls.__name__ == "Dataset1":
            self.assertTupleEqual(data['p2'].shape, (1,1))
        elif self.dataset_cls.__name__ == "Dataset2":
            self.assertTupleEqual(data['p2'].shape, (1,1,1,2)) # diagonal matrix
        self.assertTupleEqual(data['loc'].shape, (3,1))
        self.assertTupleEqual(data['csmtriu'].shape,(1,4,4,1))
        self.assertTupleEqual(data['csm'].shape,(1,4,4,2))
        self.assertTupleEqual(data['sourcemap'].shape,(1,) + dataset.grid.shape)       

    def test_multiprocessing(self):
        self.dataset.size=5
        for d in self.dataset.generate(tasks=2):
            pass

class TestDataset2(TestDataset1):

    dataset_cls = Dataset2



if __name__ == "__main__":
    unittest.main()            
