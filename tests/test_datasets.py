import shutil
import tempfile
import unittest
from os import environ, path

import numpy as np
from parameterized import parameterized

environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # change tensorflow log level for doc purposes

from acoupipe.dataset1 import DEFAULT_GRID as DEFAULT_GRID1
from acoupipe.dataset1 import DEFAULT_MICS as DEFAULT_MICS1
from acoupipe.dataset1 import Dataset1
from acoupipe.dataset2 import DEFAULT_GRID as DEFAULT_GRID2
from acoupipe.dataset2 import DEFAULT_MICS as DEFAULT_MICS2
from acoupipe.dataset2 import Dataset2

dirpath = path.dirname(path.abspath(__file__))

DEFAULT_MICS1.mpos_tot = np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])
DEFAULT_MICS2.mpos_tot = DEFAULT_MICS1.mpos_tot.copy()

DEFAULT_GRID1.increment = 1/5*1.5
DEFAULT_GRID2.increment = 1/5*1.5

feature_test_params = [
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
    ]

class TestDataset1(unittest.TestCase):
   
    dataset_cls = Dataset1

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset = self.dataset_cls(
                            size=1,
                            f=1000,
                            features=[])      
        self.cls_name = type(self.dataset).__name__

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @parameterized.expand([
        ["sourcemap", 1],
        ["csmtriu", 1],
        ["csm", 1],
        ["eigmode", 1],
        ["sourcemap", 2],
        ["csmtriu", 2],
        ["csm", 2],
        ["eigmode", 2]        
    ])   
    def test_generate(self, feature, tasks):
        """Test generate method of the datasets in single and multi task mode."""
        self.dataset.features=[feature]
        data = next(self.dataset.generate(
            split="training", startsample=20, tasks=tasks,
            cache_csm=True, cache_bf=True, cache_dir=self.test_dir, progress_bar=False))
        self.assertIn(feature, data.keys())
        # test_loc = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_loc.npy"))
        # test_p2 =  np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_p2.npy"))
        # test_feature = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_{feature}.npy"))
        # np.testing.assert_allclose(test_loc,data["loc"])
        # np.testing.assert_allclose(test_p2,data["p2"])
        # np.testing.assert_allclose(test_feature,data[feature])

    #TODO: test sample option True/False (wishart_sampling, mic_pos_sampling, ...)

    def test_multiprocessing_output_equal_single_processing(self):
        self.dataset.size = 5
        self.dataset.features = ["sourcemap","csm","csmtriu","eigmode"]
        data_gen = self.dataset.generate(split="training", tasks=1, progress_bar=False)
        data = [d for d in data_gen]
        data_gen = self.dataset.generate(split="training", tasks=3, progress_bar=False)
        data_dist = [d for d in data_gen]
        for d in data_dist:
            i = d["idx"] - 1 # idx starts at 1          
            for feature in ["sourcemap","csm","csmtriu","eigmode","p2","loc"]:
                self.assertEqual((d[feature] - data[i][feature]).sum(), 0.0)

    def test_save_tfrecord(self):
        self.dataset.features=["sourcemap","csm","csmtriu","eigmode"]
        self.dataset.save_tfrecord(split="training", name=path.join(self.test_dir,"test.h5"))

    @parameterized.expand([
        ["sourcemap"],
        ["csmtriu"],
        ["csm"],
        ["eigmode"]
    ])   
    def test_save_h5(self, feature):
        self.dataset.features=[feature]
        self.dataset.save_h5(split="training",name=path.join(self.test_dir,"test.h5"), progress_bar=False)
        # with h5py.File(path.join(self.test_dir,"test.h5"),"r") as file:
        #     data = file[f"1/{feature}"][:]
        #     test_feature = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_{feature}.npy"))
        #     np.testing.assert_allclose(test_feature,data)

    # @parameterized.expand(feature_test_params)  
    # def test_shapes(self, feature, f, num):
    #     self.dataset.features=[feature]
    #     self.dataset.num = num
    #     self.dataset.f = f
    #     data = next(self.dataset.generate())
    #     ndim = 4 # number of sources at this sample
    #     if f is None:
    #         fdim = self.dataset.freq_data.fftfreq().shape[0]
    #     elif type(f) == list:
    #         fdim = len(f)
    #     else:
    #         fdim = 1
    #     if self.dataset_cls.__name__ == "Dataset1":
    #         self.assertTupleEqual(data["p2"].shape, (fdim,ndim))
    #     elif self.dataset_cls.__name__ == "Dataset2":
    #         self.assertTupleEqual(data["p2"].shape, (fdim,ndim,ndim,2)) # diagonal matrix
    #     self.assertTupleEqual(data["loc"].shape, (3,ndim))
    #     if feature == "csm":
    #         self.assertTupleEqual(data[feature].shape, (fdim,4,4,2))
    #     elif feature == "csmtriu":
    #         self.assertTupleEqual(data[feature].shape, (fdim,4,4,1))
    #     else:
    #         self.assertTupleEqual(data[feature].shape,(fdim,) + self.dataset.grid.shape)       

    # @parameterized.expand(feature_test_params)  
    # def test_tf_dataset_shapes(self, feature, f, num):
    #     self.dataset.features=[feature]
    #     self.dataset.num = num
    #     self.dataset.f = f
    #     dataset = iter(self.dataset.get_tf_dataset())
    #     data = next(dataset)
    #     ndim = 4 # number of sources at this sample
    #     if f is None:
    #         fdim = self.dataset.freq_data.fftfreq().shape[0]
    #     elif type(f) == list:
    #         fdim = len(f)
    #     else:
    #         fdim = 1
    #     if self.dataset_cls.__name__ == "Dataset1":
    #         self.assertTupleEqual(tuple(data["p2"].shape), (fdim,ndim))
    #     elif self.dataset_cls.__name__ == "Dataset2":
    #         self.assertTupleEqual(tuple(data["p2"].shape), (fdim,ndim,ndim,2)) # diagonal matrix
    #     self.assertTupleEqual(tuple(data["loc"].shape), (3,ndim))
    #     if feature == "csm":
    #         self.assertTupleEqual(tuple(data[feature].shape), (fdim,4,4,2))
    #     elif feature == "csmtriu":
    #         self.assertTupleEqual(tuple(data[feature].shape), (fdim,4,4,1))
    #     else:
    #         self.assertTupleEqual(tuple(data[feature].shape),(fdim,) + self.dataset.grid.shape)      



class TestDataset2(TestDataset1):

    dataset_cls = Dataset2



if __name__ == "__main__":
    unittest.main()            
