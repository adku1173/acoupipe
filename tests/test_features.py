
import tempfile
from pathlib import Path

import acoular as ac
import numpy as np
import pytest

from acoupipe.datasets.base import DatasetBase
from acoupipe.datasets.features import (
    CSMDiagonalAnalytic,
    CSMFeature,
    EigmodeFeature,
    LocFeature,
    SourcemapFeature,
    SpectrogramFeature,
    TimeDataFeature,
)
from acoupipe.datasets.precision import NUMPY_COMPLEX_DTYPES, NUMPY_FLOAT_DTYPES, TF_COMPLEX_DTYPES, TF_FLOAT_DTYPES
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic, TransferMonopole

mics = ac.MicGeom(
    mpos_tot = np.random.normal(size=2*3).reshape(3,2),
)
tranfer = TransferMonopole(
    mics = mics,
    grid = ac.RectGrid(x_min=-1, x_max=1, y_min=-1, y_max=1, z=1, increment=1.0),
)
power_spectra = PowerSpectraAnalytic(
    cached = False,
    Q = np.eye(tranfer.grid.size, dtype=np.complex128)[np.newaxis],
    transfer = tranfer,
    numsamples = 512,
    block_size = 128,
    sample_freq = 51200
)

class TestFeatureBase:

    dataset = DatasetBase()
    feature = None

    def generate(self):
        return next(
            self.dataset.generate(
                size=1, split="training", features=[self.feature]))[self.feature.name]

    def generate_tf(self):
        return next(iter(
            self.dataset.get_tf_dataset(
                size=1, split="training", features=[self.feature])))[self.feature.name]

    def save_tf(self):
        test_dir = Path(tempfile.mkdtemp()) / "test.tfrecord"
        self.dataset.save_tfrecord(name=test_dir, size=1, split="training", features=[self.feature])

    def save_h5(self):
        test_dir = Path(tempfile.mkdtemp()) / "test.h5"
        self.dataset.save_h5(name=test_dir, size=1, split="training", features=[self.feature])

class TestTimeDataFeature(TestFeatureBase):

    feature = TimeDataFeature(
            time_data = ac.SineGenerator(sample_freq=51200, numsamples=2),
            shape=(None,1))

    @pytest.mark.parametrize("dtype", NUMPY_FLOAT_DTYPES.values)
    def test_dtype(self, dtype):
        self.feature.dtype = dtype
        assert self.generate().dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", NUMPY_FLOAT_DTYPES.values)
    def test_save_h5(self, dtype):
        self.feature.dtype = dtype
        self.save_h5()

    @pytest.mark.parametrize("dtype",TF_FLOAT_DTYPES.values)
    def test_tf_dtype(self, dtype):
        self.feature.tf_dtype = dtype
        assert self.generate_tf().dtype == dtype

    @pytest.mark.parametrize("dtype",TF_FLOAT_DTYPES.values)
    def test_save_tf(self, dtype):
        self.feature.tf_dtype = dtype
        self.save_tf()

    def test_values(self):
        comp_data = self.feature.time_data.signal()[:,np.newaxis]
        np.testing.assert_array_almost_equal(
            self.generate(), comp_data)


class TestLocFeature(TestTimeDataFeature):

    feature = LocFeature(
        freq_data = power_spectra
    )

    def test_values(self):
        comp_data = self.feature.freq_data.transfer.grid.gpos
        np.testing.assert_array_almost_equal(
            self.generate(), comp_data)


class TestSourcemapFeature(TestTimeDataFeature):

    feature = SourcemapFeature(
        beamformer = ac.BeamformerBase(
            freq_data=power_spectra,
            cached=False,
            steer=ac.SteeringVector(mics=mics, grid=power_spectra.transfer.grid),
    ))

    def test_values(self):
        pass

class TestCSMDiagonalAnalytic(TestTimeDataFeature):

    feature = CSMDiagonalAnalytic(
        freq_data = power_spectra,
    )

    def test_values(self):
        pass


class TestSpectrogramFeature(TestFeatureBase):

    feature = SpectrogramFeature(
            freq_data = ac.FFTSpectra(block_size=128,
                source=ac.TimeSamples(data=np.random.rand(128, 1), sample_freq=51200),
            ))

    @pytest.mark.parametrize("dtype", NUMPY_COMPLEX_DTYPES.values)
    def test_dtype(self, dtype):
        self.feature.dtype = dtype
        assert self.generate().dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", NUMPY_COMPLEX_DTYPES.values)
    def test_save_h5(self, dtype):
        self.feature.dtype = dtype
        self.save_h5()

    @pytest.mark.parametrize("dtype", TF_COMPLEX_DTYPES.values)
    def test_tf_dtype(self, dtype):
        self.feature.tf_dtype = dtype
        assert self.generate_tf().dtype == dtype

    @pytest.mark.parametrize("dtype", TF_COMPLEX_DTYPES.values)
    def test_save_tf(self, dtype):
        self.feature.tf_dtype = dtype
        self.save_tf()

    # def test_values(self): # need to be fixed in Acoular! (should consider different frequencies)
    #     comp_data = ac.tools.return_result(self.feature.freq_data)
    #     np.testing.assert_array_almost_equal(
    #         self.generate(), comp_data)


class TestCSMFeature(TestSpectrogramFeature):

    feature = CSMFeature(
        freq_data = power_spectra,
    )

    def test_values(self):
        comp_data = self.feature.freq_data.csm[:]
        np.testing.assert_array_almost_equal(
            self.generate(), comp_data)

class TestEigmodeFeature(TestCSMFeature):

    feature = EigmodeFeature(
        freq_data = power_spectra,
    )

    def test_values(self):
        pass



class CSMtriuFeature(TestTimeDataFeature):

    def test_values(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
