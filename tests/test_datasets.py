import unittest
from os import path
from parameterized import parameterized
from acoupipe.dataset1 import Dataset1, config1
from acoupipe.dataset2 import Dataset2, config2
import numpy as np
import h5py
import tempfile
import shutil

dirpath = path.dirname(path.abspath(__file__))

class TestDataset1(unittest.TestCase):
   
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config1['increment'] = 1/5*1.5
        config1['ref_mic'] = 0
        config1['blocksize'] = 128
        self.dataset = Dataset1(split="training",
                            size=1,
                            startsample=20,
                            features=["csm"],
                            config=config1)        
        self.dataset.mics.invalid_channels = [i for i in range(60)]
        self.dataset.noisy_mics.invalid_channels = [i for i in range(60)]
        self.cls_name = type(self.dataset).__name__

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @parameterized.expand([
        ["sourcemap"],
        ["csmtriu"],
        ["sourcemap"],
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
        ["sourcemap"],
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
    def test_freq_num(self, feature, f, num):
        self.dataset.features=[feature]
        self.dataset.num = num
        self.dataset.f = f
        data = next(self.dataset.generate())


class TestDataset2(TestDataset1):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        config2['increment'] = 1/5*1.5
        config2['ref_mic'] = 0
        config2['blocksize'] = 128
        self.dataset = Dataset2(split="training",
                            size=1,
                            startsample=20,
                            features=["csm"],
                            config=config2)        
        self.dataset.mics.invalid_channels = [i for i in range(60)]
        self.dataset.noisy_mics.invalid_channels = [i for i in range(60)]
        self.cls_name = type(self.dataset).__name__


if __name__ == "__main__":
    unittest.main()            
