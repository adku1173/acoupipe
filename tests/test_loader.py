from functools import partial

from acoupipe import loader as loader_module
from acoupipe.loader import LoadH5Dataset

import pytest
from h5py import File as H5File


def test_load_h5_data(h5_test_file):
    """Test loading data from an HDF5 file."""
    ds = LoadH5Dataset(name=h5_test_file)
    assert ds.numsamples == 5
    assert ds.basename == 'test_data'
    assert ds.h5f['1']['data'][()]


@pytest.mark.parametrize('initialize', ['constructor', 'name'])
@pytest.mark.parametrize(
    ('method', 'features'),
    [
        ('get_data', None),
        ('get_dataset_generator', None),
        ('get_dataset_generator', ['value']),
        ('get_dataset_generator', []),
    ],
)
def test_read_samples_in_numeric_order(initialize, method, features, tmp_path):
    """Both readers preserve numeric ordering and top-level feature selection."""
    filename = str(tmp_path / 'ordered.h5')
    with H5File(filename, 'w') as file:
        for index in (10, 2, 1):
            sample = file.create_group(str(index))
            sample['value'] = index
            sample['extra'] = True
        file.create_group('metadata')['version'] = 1
    if initialize == 'constructor':
        loader = LoadH5Dataset(name=filename)
    else:
        loader = LoadH5Dataset()
        loader.name = filename
    with loader.h5f:
        samples = list(loader.get_data()) if method == 'get_data' else list(loader.get_dataset_generator(features)())
        assert [sample['idx'] for sample in samples] == ['1', '2', '10']
        for sample in samples:
            assert set(sample) == ({'idx', 'value', 'extra'} if features is None else {'idx', *features})
            if features is None or 'value' in features:
                assert sample['value'] == int(sample['idx'])


@pytest.mark.parametrize('method', ['constructor', 'load_data'])
@pytest.mark.parametrize(('invalid', 'error'), [('index', ValueError), ('metadata', TypeError)])
def test_load_errors_propagate_and_close_file(method, invalid, error, tmp_path, monkeypatch):
    """Malformed files raise their original error without leaking the open file."""
    filename = str(tmp_path / 'invalid.h5')
    with H5File(filename, 'w') as file:
        file.create_group('invalid' if invalid == 'index' else '1')
        if invalid == 'metadata':
            file['metadata'] = 1
    opened = []

    def open_file(*args, **kwargs):
        file = H5File(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(loader_module, 'H5File', open_file)
    if method == 'constructor':
        load = partial(LoadH5Dataset, name=filename)
    else:
        loader = LoadH5Dataset()
        loader.trait_setq(name=filename)
        load = loader.load_data
    with pytest.raises(error):
        load()
    assert len(opened) == 1
    assert not opened[0].id.valid
