from os import environ

environ["NUMBA_NUM_THREADS"] = "1" # set numba threads to 1
environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # change tensorflow log level for doc purposes
import shutil
import tempfile
import unittest
from pathlib import Path

import acoular as ac
import numpy as np
import tensorflow as tf
from parameterized import parameterized

from acoupipe.datasets.experimental import DatasetMIRACLE, DatasetMIRACLEConfig
from acoupipe.datasets.synthetic import Dataset1TestConfig, DatasetSynthetic1

IMPLEMENTED_FEATURES = ["time_data","csm","csmtriu","sourcemap","eigmode", "spectrogram"] + [
    "seeds", "idx","loc","source_strength_analytic", "source_strength_estimated", "noise_strength_analytic",
    "noise_strength_estimated","f","num"]

dirpath = Path(__file__).parent.absolute()
modes = [["welch"], ["analytic"], ["wishart"]]
validation_data_path = Path(__file__).parent.absolute() / "validation_data"
start_idx=3
tasks=2

#TODO: speed up tests

class TestDatasetSynthetic1(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        #print(f"Creating {self.test_dir}")

    @staticmethod
    def create_dataset(full=False, tasks=1, **kwargs):
        if full:
            return DatasetSynthetic1(tasks=tasks,**kwargs)
        config = Dataset1TestConfig(**kwargs)
        return DatasetSynthetic1(config=config,tasks=tasks,**kwargs)

    def tearDown(self):
        #print(f"Removing {self.test_dir}")
        shutil.rmtree(self.test_dir)

    @parameterized.expand(modes)
    def test_values_correct(self,mode):
        """Test generate method of the datasets in single task mode."""
        for feature in IMPLEMENTED_FEATURES:
            if mode == "analytic" and "_estimated" in feature:
                continue
            if mode != "welch" and feature in ["spectrogram","time_data"]:
                continue
            for f in [None,1000]:
                for num in [0,3]:
                    if f is None and num != 0:
                        continue
                    with self.subTest(f"feature={feature}, f={f}, num={num}"):
                        dataset = self.create_dataset(mode=mode)
                        gen = dataset.generate(split="training",progress_bar=False, size=10000,start_idx=start_idx,
                        f=f,num=num,features=[feature])
                        while True:
                            data = next(gen)
                            if data["idx"] == start_idx:
                                break
                        test_data = np.load(validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy")
                        np.testing.assert_allclose(data[feature],test_data,rtol=1e-5, atol=1e-8)

    @parameterized.expand(modes)
    def test_multiprocessing_values_correct(self,mode):
        """Test generate method of the datasets in single task mode."""
        for feature in IMPLEMENTED_FEATURES:
            if mode == "analytic" and "_estimated" in feature:
                continue
            if mode != "welch" and feature in ["spectrogram","time_data"]:
                continue
            for f in [1000]:
                for num in [0]:
                    if f is None and num != 0:
                        continue
                    with self.subTest(f"feature={feature}, f={f}, num={num}"):
                        #ray.shutdown()
                        #ray.init(log_to_driver=False)
                        dataset = self.create_dataset(mode=mode,tasks = tasks)
                        gen = dataset.generate(
                            split="training",progress_bar=False, size=100,start_idx=1,f=f,num=num,features=[feature],)
                        while True:
                            data = next(gen)
                            if data["idx"] == start_idx:
                                break
                        test_data = np.load(
                            validation_data_path / f"{type(dataset).__name__}_{feature}_f{f}_num{num}_mode{mode}.npy")
                        np.testing.assert_allclose(data[feature],test_data,rtol=1e-5, atol=1e-8)

    @parameterized.expand(modes)
    def test_save_tfrecord(self, mode):
        for feature in IMPLEMENTED_FEATURES:
            for mic_sig_noise in [True, False]:
                with self.subTest(f"feature={feature}, mic_sig_noise={mic_sig_noise}"):
                    if mode == "analytic" and "_estimated" in feature:
                        continue
                    if mode != "welch" and feature in ["spectrogram","time_data"]:
                        continue
                    dataset = self.create_dataset(mode, mic_sig_noise=mic_sig_noise)
                    dataset.save_tfrecord(split="training", size=2,features=[feature],
                        name=self.test_dir / "test.tfrecord", progress_bar=False)

    @parameterized.expand(modes)
    def test_parse_tfrecord(self, mode):
        for feature in IMPLEMENTED_FEATURES + ["idx", "seeds"]:
            for mic_sig_noise in [True, False]:
                for num in [0,3]:
                    for f in [None, 1000]:
                        if num == 3 and f is None:
                            continue
                        if mode == "analytic" and "estimated" in feature:
                            continue
                        if mode != "welch" and feature in ["spectrogram","time_data"]:
                            continue
                        with self.subTest(f"feature={feature}, f={f}, num={num}, mic_sig_noise={mic_sig_noise}"):
                            # generate data
                            dataset = self.create_dataset(mode=mode, mic_sig_noise=mic_sig_noise)
                            data_generated = next(dataset.generate(
                                f=f, num=num, split="training", size=1, progress_bar=False,features=[feature]))
                            # save and parse data
                            dataset.save_tfrecord(
                                f=f, num=num, split="training", size=1, progress_bar=False,features=[feature],
                                name=self.test_dir / "test.tfrecord")
                            parser = dataset.get_tfrecord_parser(f=f,num=num,features=[feature])
                            tfrecord = tf.data.TFRecordDataset(self.test_dir / "test.tfrecord").map(parser)
                            data_loaded = next(iter(tfrecord))
                            # compare data
                            np.testing.assert_allclose(data_generated[feature],data_loaded[feature],rtol=1e-5, atol=1e-8)

    @parameterized.expand(modes)
    def test_save_h5(self, mode):
        for feature in IMPLEMENTED_FEATURES:
            with self.subTest(f"feature={feature}"):
                if mode == "analytic" and "_estimated" in feature:
                    continue
                if mode != "welch" and feature in ["spectrogram","time_data"]:
                    continue
                dataset = self.create_dataset(mode)
                dataset.save_h5(split="training", size=2, features=[feature],
                    name=self.test_dir / "test.h5", progress_bar=False)


    @parameterized.expand(modes)
    def test_get_feature_shapes(self, mode):
        """Test if the output of the get_feature_shapes method is matches with the generated shapes.

        This test consideres a fixed number of sources (varying numbers result in None type shapes, which
        cannot be compared with the generated shapes). Vaying source numbers are implicitly tested in the
        test_get_tf_dataset method.
        """
        for feature in IMPLEMENTED_FEATURES:
            for num in [0,3]:
                for f in [None, 1000]:
                    if num == 3 and f is None:
                        continue
                    with self.subTest(f"feature={feature}, f={f}, num={num}"):
                        if mode == "analytic" and "_estimated" in feature:
                            continue
                        if mode != "welch" and feature in ["spectrogram","time_data"]:
                            continue
                        dataset = self.create_dataset(mode)
                        data = next(dataset.generate(f=f, num=num, features=[feature],
                                                split="training", size=1, progress_bar=False))
                        feature_collection = dataset.get_feature_collection(features=[feature],f=f,num=num)
                        desired_shape = feature_collection.feature_tf_shape_mapper[feature]
                        for i in range(len(desired_shape)):
                            if desired_shape[i] is not None:
                                self.assertEqual(desired_shape[i], data[feature].shape[i])

    @parameterized.expand(modes)
    def test_get_tf_dataset(self,mode):
        """Test if a tensorflow dataset can be constructed from the pipeline."""
        for feature in IMPLEMENTED_FEATURES:
            for num in [0,3]:
                for f in [None, 1000]:
                    if num == 3 and f is None:
                        continue
                    for mic_sig_noise in [True, False]:
                        with self.subTest(f"feature={feature}, f={f}, num={num}, mic_sig_noise={mic_sig_noise}"):
                            if mode == "analytic" and "_estimated" in feature:
                                continue
                            if mode != "welch" and feature in ["spectrogram","time_data"]:
                                continue
                            dataset = self.create_dataset(
                                mode,mic_sig_noise=mic_sig_noise)
                            dataset = dataset.get_tf_dataset(split="training", size=1, progress_bar=False,
                             f=f, num=num, features=[feature],)
                            data = next(iter(dataset))
                            self.assertIn(feature, data.keys())

    @parameterized.expand(modes)
    def test_toggle_sampler(self,mode):
        arguments = [{"mic_pos_noise" : True}, {"mic_pos_noise" : False},
                    {"mic_sig_noise" : True}, {"mic_sig_noise" : False},]
        for kwargs in arguments:
            with self.subTest(f"kwargs={kwargs}"):
                dataset = self.create_dataset(mode=mode, **kwargs)
                gen = dataset.generate(split="training", size=1,progress_bar=False,
                features=["csm","noise_strength_estimated","noise_strength_analytic"],)
                data = next(gen)
                self.assertIn("csm", data.keys())
                if dataset.config.mic_sig_noise is not False:
                    if mode != "analytic":
                        self.assertGreater(data["noise_strength_estimated"].sum(), 0)
                    self.assertGreater(data["noise_strength_analytic"].sum(), 0)
                else:
                    if mode != "analytic":
                        self.assertEqual(data["noise_strength_estimated"].sum(), 0)
                    self.assertEqual(data["noise_strength_analytic"].sum(), 0)

    @parameterized.expand(modes)
    def test_csm_prmssq(self, mode):
        features = ["csm","source_strength_estimated","source_strength_analytic","noise_strength_estimated","noise_strength_analytic"]
        for mic_sig_noise in [True, False]:
            for num in [0,3]:
                for f in [None, 1000]:
                    if num == 3 and f is None:
                        continue
                    with self.subTest(f"f={f}, num={num}, mic_sig_noise={mic_sig_noise}"):
                        dataset = self.create_dataset(full=True, mode=mode,
                             mic_sig_noise=mic_sig_noise,mic_pos_noise=False)
                        gen = dataset.generate(
                            f=f, num=num, features=features,
                            split="training", size=1, progress_bar=False)
                        data = next(gen)
                        csm_psq = data["csm"][:,63,63]
                        if mode != "analytic":
                            noise_psq = data["noise_strength_estimated"][:,63]
                            sig_psq = data["source_strength_estimated"].sum(1)
                        else:
                            noise_psq = data["noise_strength_analytic"][:,63]
                            sig_psq = data["source_strength_analytic"].sum(1)
                        if mode != "analytic":
                            np.testing.assert_allclose(csm_psq, noise_psq + sig_psq,rtol=1e-1, atol=1e-1)
                        else:
                            np.testing.assert_allclose(csm_psq, noise_psq + sig_psq,rtol=1e-5, atol=1e-8)

    @parameterized.expand(modes)
    def test_sourcemap_max(self, mode):
        """A plausability test_."""
        features = ["sourcemap","source_strength_estimated","source_strength_analytic"]
        for num in [0,3]:
            for f in [4000]:
                with self.subTest(f"f={f}, num={num}"):
                    dataset = self.create_dataset(full=False, mode=mode,
                            mic_sig_noise=False,mic_pos_noise=False,
                            max_nsources=1, snap_to_grid=True)
                    dataset.config.fft_params["block_size"] = 512
                    gen = dataset.generate(
                        f=f, num=num, features=features,
                        split="training", size=1, progress_bar=False)
                    data = next(gen)
                    sourcemap_max = ac.L_p(data["sourcemap"].max())
                    source_stength_estimated = ac.L_p(data["source_strength_estimated"].max())
                    np.testing.assert_allclose(sourcemap_max,source_stength_estimated,atol=1e-1)

    @parameterized.expand(modes)
    def test_eigvalsum_equal_csm(self, mode):
        """A plausability test_."""
        features = ["csm","eigmode"]
        for num in [0]:
            for f in [4000]:
                with self.subTest(f"f={f}, num={num}"):
                    dataset = self.create_dataset(full=True, mode=mode,
                            mic_sig_noise=False,mic_pos_noise=False,
                            max_nsources=1, snap_to_grid=True)
                    dataset.config.fft_params["block_size"] = 512
                    gen = dataset.generate(
                        f=f, num=num, features=features,
                        split="training", size=1, progress_bar=False)
                    data = next(gen)
                    eig, eigvec = np.linalg.eigh(data["csm"][0])
                    eig_eig = np.linalg.norm(data["eigmode"][0],axis=0)
                    np.testing.assert_allclose(eig_eig,eig,rtol=1e-5, atol=1e-7)



class MIRACLEDataset(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        #print(f"Creating {self.test_dir}")

    def tearDown(self):
        #print(f"Removing {self.test_dir}")
        shutil.rmtree(self.test_dir)

    def create_dataset(self, full=True, tasks=1, **kwargs):
        srir_dir = Path("/tmp/")
        config = DatasetMIRACLEConfig(srir_dir=srir_dir, **kwargs)
        return DatasetMIRACLE(config=config,tasks=tasks,**kwargs)


    @parameterized.expand(modes)
    def test_csm_prmssq(self, mode):
        features = ["csm","source_strength_estimated","source_strength_analytic","noise_strength_estimated","noise_strength_analytic"]
        for mic_sig_noise in [True, False]:
            for num in [0,3]:
                for f in [None, 1000]:
                    if num == 3 and f is None:
                        continue
                    with self.subTest(f"f={f}, num={num}, mic_sig_noise={mic_sig_noise}"):
                        dataset = self.create_dataset(full=True, mode=mode,mic_sig_noise=mic_sig_noise, max_nsources=1)
                        gen = dataset.generate(
                            f=f, num=num, features=features,
                            split="training", size=1, progress_bar=False)
                        data = next(gen)
                        csm_psq = data["csm"][:,63,63].sum()
                        if mode != "analytic":
                            noise_psq = data["noise_strength_estimated"][:,63].sum()
                            sig_psq = data["source_strength_estimated"][:].sum()
                        else:
                            noise_psq = data["noise_strength_analytic"][:,63].sum()
                            sig_psq = data["source_strength_analytic"][:].sum()
                        if mode != "analytic":
                            self.assertAlmostEqual(csm_psq, noise_psq + sig_psq, places=1)
                        else:
                            np.testing.assert_allclose(csm_psq, noise_psq + sig_psq)

    @parameterized.expand(modes)
    def test_sourcemap_max(self, mode):
        """A plausability test. Tolerance is large -> loudspeaker not a perfect monopole."""
        features = ["sourcemap","source_strength_estimated","source_strength_analytic"]
        for num in [0,3]:
            for f in [1000]:
                with self.subTest(f"f={f}, num={num}"):
                    dataset = self.create_dataset(full=True, mode=mode, max_nsources=1,
                            mic_sig_noise=False)
                    gen = dataset.generate(
                        f=f, num=num, features=features,
                        split="training", size=1, progress_bar=False)
                    data = next(gen)
                    sourcemap_max = ac.L_p(data["sourcemap"].max())
                    source_stength_estimated = ac.L_p(data["source_strength_estimated"].max())
                    np.testing.assert_allclose(sourcemap_max,source_stength_estimated,atol=1e0)

if __name__ == "__main__":

    unittest.main()
