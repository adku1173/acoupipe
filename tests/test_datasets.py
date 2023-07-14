import shutil
import tempfile
import unittest
from os import environ, path

import numpy as np
import ray
from parameterized import parameterized

from acoupipe.datasets.dataset1 import DEFAULT_GRID as DEFAULT_GRID1
from acoupipe.datasets.dataset1 import DEFAULT_MICS as DEFAULT_MICS1
from acoupipe.datasets.dataset1 import Dataset1
from acoupipe.datasets.dataset2 import DEFAULT_GRID as DEFAULT_GRID2
from acoupipe.datasets.dataset2 import DEFAULT_MICS as DEFAULT_MICS2
from acoupipe.datasets.dataset2 import Dataset2

environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # change tensorflow log level for doc purposes

dirpath = path.dirname(path.abspath(__file__))

DEFAULT_MICS1.mpos_tot = np.array([[-0.68526741, -0.7593943 , -1.99918406,  0.08414458],
       [-0.60619132,  1.20374544, -0.27378946, -1.38583541],
       [ 0.32909911,  0.56201909, -0.24697204, -0.68677001]])
DEFAULT_MICS2.mpos_tot = DEFAULT_MICS1.mpos_tot.copy()

DEFAULT_GRID1.increment = 1/5*1.5
DEFAULT_GRID2.increment = 1/5*1.5

class TestDataset1(unittest.TestCase):

    dataset_cls = Dataset1

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset = self.dataset_cls(
                            f=1000,
                            features=[])
        self.cls_name = type(self.dataset).__name__

    def tearDown(self):
        ray.shutdown()
        shutil.rmtree(self.test_dir)

    def test_values_correct(self):
        """Test generate method of the datasets in single and multi task mode."""
        for feature in ["sourcemap","csmtriu","csm","eigmode"]:
            with self.subTest(f"{feature}"):
                self.dataset.features=[feature]
                data = next(self.dataset.generate(
                    split="training", size=1, tasks=1,
                    cache_csm=False, cache_bf=False, progress_bar=False))
                test_loc = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_loc.npy"))
                test_p2 =  np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_p2.npy"))
                test_feature = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_{feature}.npy"))
                np.testing.assert_allclose(test_loc,data["loc"])
                np.testing.assert_allclose(test_p2,data["p2"])
                np.testing.assert_allclose(test_feature,data[feature])

    def test_multiprocessing_output_equal_single_processing(self):
        self.dataset.features = ["sourcemap","csm","csmtriu","eigmode"]
        data_gen = self.dataset.generate(split="training", size = 5, tasks=1, progress_bar=False)
        data = [d for d in data_gen]
        data_gen = self.dataset.generate(split="training", size = 5, tasks=3, progress_bar=False)
        data_dist = [d for d in data_gen]
        for d in data_dist:
            i = d["idx"] - 1 # idx starts at 1
            for feature in ["sourcemap","csm","csmtriu","eigmode","p2","loc"]:
                self.assertEqual((d[feature] - data[i][feature]).sum(), 0.0)

    def test_save_tfrecord(self):
        self.dataset.features=["sourcemap","csm","csmtriu","eigmode"]
        self.dataset.save_tfrecord(split="training", size=1, name=path.join(self.test_dir,"test.h5"), progress_bar=False)

    def test_save_h5(self):
        self.dataset.features=["sourcemap","csm","csmtriu","eigmode"]
        self.dataset.save_h5(split="training", size=1, name=path.join(self.test_dir,"test.h5"), progress_bar=False)

    @parameterized.expand([
        [None, 0],
        [None, 3],
        [1000, 0],
        [1000, 3],
    ])
    def test_get_feature_shapes(self, f, num):
        """Test if the output of the get_feature_shapes method is matches with the generated shapes.

        This test consideres a fixed number of sources (varying numbers result in None type shapes, which
        cannot be compared with the generated shapes). Vaying source numbers are implicitly tested in the
        test_get_tf_dataset method.
        """
        self.dataset.max_nsources = 3
        self.dataset.min_nsources = 3
        self.dataset.f = f
        self.dataset.num = num
        self.dataset.features=["sourcemap","csm","csmtriu","eigmode"]
        desired_shapes = self.dataset.get_feature_shapes()
        data = next(self.dataset.generate(split="training", size=1, progress_bar=False))
        generated_shapes = {k:v.shape for k,v in data.items() if k not in ["idx"]}
        desired_shapes = {k:v for k,v in desired_shapes.items() if k not in ["idx"]}
        self.assertDictEqual(desired_shapes, generated_shapes)

    @parameterized.expand([
        [None, 0],
        [None, 3],
        [1000, 0],
        [1000, 3],
    ])
    def test_get_tf_dataset(self, f, num):
        """Test if a tensorflow dataset can be constructed from the pipeline."""
        features = ["sourcemap","csm","csmtriu","eigmode"]
        self.dataset.f = f
        self.dataset.num = num
        self.dataset.features=features
        dataset = self.dataset.get_tf_dataset(split="training", size=1, tasks=1, progress_bar=False)
        data = next(iter(dataset))
        for f in features+["p2","loc","idx"]:
            self.assertIn(f, data.keys())



class TestDataset2(TestDataset1):

    dataset_cls = Dataset2

    def test_values_correct(self):
        """Test generate method of the datasets in single and multi task mode."""
        for sample_noise in [True, False]:
            for sample_wishart in [False, True]:
                for sample_spectra in [True, False]:
                    self.dataset.features=["sourcemap","csmtriu","csm", "eigmode"]
                    self.dataset.sample_wishart = sample_wishart
                    self.dataset.sample_noise = sample_noise
                    self.dataset.sample_spectra = sample_spectra
                    data = next(self.dataset.generate(
                        split="training", tasks=1, size=1, progress_bar=False))
                    for feature in ["sourcemap","csmtriu","csm", "eigmode"]:
                        with self.subTest(
                        f"feature={feature}, sample_wishart={sample_wishart},sample_noise={sample_noise}, sample_spectra={sample_spectra}"):
                            test_loc = np.load(path.join(dirpath,"validation_data",f"{self.cls_name}_loc.npy"))
                            test_p2 =  np.load(path.join(dirpath,"validation_data",
                                                f"{self.cls_name}_p2_wishart{sample_wishart}_noise{sample_noise}_spectra{sample_spectra}.npy"))
                            test_feature = np.load(path.join(dirpath,"validation_data",
                                        f"{self.cls_name}_{feature}_wishart{sample_wishart}_noise{sample_noise}_spectra{sample_spectra}.npy"))
                            np.testing.assert_allclose(test_loc,data["loc"])
                            np.testing.assert_allclose(test_p2,data["p2"],1e-8)
                            np.testing.assert_allclose(test_feature,data[feature],1e-8)

    def test_plausibility(self):
        """Test if the generated data is plausible.

        * Variance of the sources should match the sum of the frequency amplitudes.
        * sum of the p2 values should equal the variances (in full frequency mode).
        * sum of the csm auto-power should be 1.
        """
        for sample_spectra in [True,False]:
            with self.subTest(f"sample spectra {sample_spectra}"):
                self.dataset.f = None
                self.dataset.sample_spectra = sample_spectra
                self.dataset.features=["csm"]
                data = next(self.dataset.generate(split="training", size=1, progress_bar=False))
                np.testing.assert_allclose(np.diagonal(data["p2"][...,0].sum(0)), data["variances"])
                # ref_mic_idx = np.argmin(
                #     np.linalg.norm((self.dataset.steer.mics.mpos - self.dataset.steer.mics.center[:,np.newaxis]),axis=0))
#                np.testing.assert_allclose(data["csm"][:,ref_mic_idx,ref_mic_idx,0].sum(), 1, atol=1e-2)

    @parameterized.expand([
        [None, 0, True, False, True],
        [1000, 0, True, False, True],
        [1000, 0, True, True, True],
        [None, 3, False, True, True],
        [1000, 3, False, True, True],
        [1000, 3, False, False, True],
        [None, 0, True, False, False],
        [1000, 0, True, False, False],
        [1000, 0, True, True, False],
        [None, 3, False, True, False],
        [1000, 3, False, True, False],
        [1000, 3, False, False, False],
    ])
    def test_get_feature_shapes(self, f, num, sample_wishart, sample_noise, sample_spectra):
        """Test if the output of the get_feature_shapes method is matches with the generated shapes.

        This test consideres a fixed number of sources (varying numbers result in None type shapes, which
        cannot be compared with the generated shapes). Vaying source numbers are implicitly tested in the
        test_get_tf_dataset method.
        """
        self.dataset.sample_spectra = sample_spectra
        self.dataset.sample_wishart = sample_wishart
        self.dataset.sample_noise = sample_noise
        self.dataset.max_nsources = 3
        self.dataset.min_nsources = 3
        self.dataset.f = f
        self.dataset.num = num
        self.dataset.features=["sourcemap","csm","csmtriu","eigmode"]
        desired_shapes = self.dataset.get_feature_shapes()
        data = next(self.dataset.generate(split="training", size=1, progress_bar=False))
        generated_shapes = {k:v.shape for k,v in data.items() if k not in ["idx"]}
        desired_shapes = {k:v for k,v in desired_shapes.items() if k not in ["idx"]}
        self.assertDictEqual(desired_shapes, generated_shapes)

    @parameterized.expand([
        [None, 0, True, False, True],
        [1000, 0, True, False, True],
        [1000, 0, True, True, True],
        [None, 3, False, True, True],
        [1000, 3, False, True, True],
        [1000, 3, False, False, True],
        [None, 0, True, False, False],
        [1000, 0, True, False, False],
        [1000, 0, True, True, False],
        [None, 3, False, True, False],
        [1000, 3, False, True, False],
        [1000, 3, False, False, False],
    ])
    def test_get_tf_dataset(self, f, num, sample_wishart, sample_noise, sample_spectra):
        """Test if a tensorflow dataset can be constructed from the pipeline."""
        features = ["sourcemap","csm","csmtriu","eigmode"]
        self.dataset.sample_spectra = sample_spectra
        self.dataset.sample_wishart = sample_wishart
        self.dataset.sample_noise = sample_noise
        self.dataset.f = f
        self.dataset.num = num
        self.dataset.features=features
        dataset = self.dataset.get_tf_dataset(split="training", size=1, tasks=1, progress_bar=False)
        data = next(iter(dataset))
        for f in features+["p2","loc","idx"]:
            self.assertIn(f, data.keys())

if __name__ == "__main__":
    unittest.main()
