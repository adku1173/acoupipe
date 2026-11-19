"""Provides classes to store the data extracted by :class:`~acoupipe.pipeline.BasePipeline`.

Purpose of the Writer Module
----------------------------
The :code:`writer.py` module provides classes to store the data extracted by the pipeline.
The current implementation includes classes to save data in a container-like file format
(.h5 file with the :code:`WriteH5Dataset` class). The TFRecord format, which can be efficiently
consumed by the Tensorflow framework for machine learning, is written by
:class:`~acoupipe.tfrecord.WriteTFRecord`.

.. code-block:: python

    file_writer = acoupipe.writer.WriteH5Dataset(
        source=pipeline,
    )

    file_writer.save()


"""

from datetime import datetime
from os import path

from acoular import config
from acoupipe.base import DataGenerator

from h5py import File as H5File
from traits.api import Bool, Dict, File, Instance, List


class BaseWriteDataset(DataGenerator):
    """Base class intended to write data to a specific file format.

    Writes data from :class:`~acoupipe.pipeline.BasePipeline` instances.

    This class has no functionality and should not be used.
    """

    #: source instance that has to be of type :class:`~acoupipe.base.DataGenerator`
    source = Instance(DataGenerator)

    def save(self):
        """Save data from the :attr:`source` instance to file."""
        # write to File...

    def get_data(self, progress_bar=True, start_idx=1):
        """Save source output data to file and pass the data to the next object.

        Parameters
        ----------
        progress_bar : bool
            If True, a progress bar is shown.
        start_idx : int
            Index of the first sample which is written to file.

        Returns
        -------
        dict
            Dictionary containing a sample of the data set
            {feature_name[key] : feature[values]}.
        """
        yield from self.source.get_data(progress_bar, start_idx)


class WriteH5Dataset(BaseWriteDataset):
    """Class intended to write data to a `.h5` file."""

    #: Name of the file to be saved.
    name = File(filter=['*.h5'], desc='name of data file')

    # #: Number of samples to write to file by :meth:`result` method.
    # #: defaults to -1 (write as long as source yields data).
    # numsamples_write = Int(-1)

    #: flag that can be raised to stop file writing when running in detached thread
    writeflag = Bool(True)

    #: a list with names of the features to be saved. By default [],
    #: meaning that all features comming from the source will be saved.
    features = List([], desc='the names of the features to be saved')

    #: Metadata to be stored in HDF5 file object
    metadata = Dict(desc='metadata to be stored in .h5 file')

    def create_filename(self):
        if self.name == '':
            name = datetime.now().isoformat('_').replace(':', '-').replace('.', '_')  # noqa: DTZ005
            self.name = path.join(config.td_dir, name + '.h5')

    def get_file(self):
        self.create_filename()
        return H5File(self.name, mode='w')

    def get_filtered_features(self):
        if self.features:
            return self.features.copy() + ['idx'] if 'idx' not in self.features else self.features
        return None

    def _add_data(self, f5h, data, subf):
        dataset_idx = str(data['idx'])
        # create a group for each Sample
        f5h.create_group(dataset_idx)
        # store dict in the group
        if not subf:
            [f5h.create_dataset(f'{dataset_idx}/{key}', data=value) for key, value in data.items()]
        else:
            [f5h.create_dataset(f'{dataset_idx}/{key}', data=value) for key, value in data.items() if key in subf]

    def _add_metadata(self, f5h):
        """Add metadata to .h5 file."""
        nitems = len(self.metadata.items())
        if nitems > 0:
            f5h.create_group('metadata')
            for key, value in self.metadata.items():
                f5h.create_dataset(f'metadata/{key}', data=value)

    def save(self, progress_bar=True, start_idx=1):
        """Save the :meth:`get_data()` output of the :attr:`source` to .h5 file format."""
        f5h = self.get_file()
        subf = self.get_filtered_features()
        for data in self.source.get_data(progress_bar, start_idx):
            self._add_data(f5h, data, subf)
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()

    def get_data(self, progress_bar=True, start_idx=1):
        """Save the source data to a `*.h5` file and yield it to the next object.

        Returns
        -------
        dict
            Dictionary containing a sample of the data set
            {feature_name[key] : feature[values]}.
        """
        self.writeflag = True
        f5h = self.get_file()
        subf = self.get_filtered_features()
        for data in self.source.get_data(progress_bar, start_idx):
            if not self.writeflag:
                return
            self._add_data(f5h, data, subf)
            yield data
            f5h.flush()
        self._add_metadata(f5h)
        f5h.flush()
        f5h.close()
