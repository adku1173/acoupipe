"""Utilities for reading and writing nested HDF5 values."""

from h5py import Group


def read_h5_value(value):
    """Read groups as nested dictionaries and datasets as their stored values."""
    if isinstance(value, Group):
        return {key: read_h5_value(item) for key, item in value.items()}
    return value[()]


def write_h5_value(group, name, value):
    """Write dictionaries as groups and other values as datasets."""
    if isinstance(value, dict):
        subgroup = group.create_group(name)
        for key, item in value.items():
            write_h5_value(subgroup, key, item)
    else:
        group.create_dataset(name, data=value)
