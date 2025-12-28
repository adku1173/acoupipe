from acoupipe.loader import LoadH5Dataset


def test_load_h5_data(h5_test_file):
    """Test loading data from an HDF5 file."""
    ds = LoadH5Dataset(name=h5_test_file)
    assert ds.numsamples == 5
    assert ds.basename == 'test_data'
    assert ds.h5f['1']['data'][()]
