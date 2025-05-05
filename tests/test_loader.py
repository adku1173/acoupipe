import os
import tempfile

import pytest

from acoupipe.loader import LoadH5Dataset
from acoupipe.writer import WriteH5Dataset

from .pipeline_value_test import get_pipeline


@pytest.fixture
def h5_test_file():
    """Fixture to create and clean up a temporary HDF5 test file."""
    pipeline = get_pipeline(5)
    pipeline.random_seeds = {
        1: range(1, 6),
        2: range(2, 7),
        3: range(3, 8),
        4: range(4, 9),
    }
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, 'test_data.h5')
    writer = WriteH5Dataset(source=pipeline, name=file_path)
    writer.save()
    yield file_path
    os.remove(file_path)
    os.rmdir(temp_dir)


def test_load_h5_data(h5_test_file):
    """Test loading data from an HDF5 file."""
    ds = LoadH5Dataset(name=h5_test_file)
    assert ds.numsamples == 5
    assert ds.basename == 'test_data'
    assert ds.h5f['1']['data'][()]
