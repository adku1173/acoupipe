.. _data:


Datasets
========

AcouPipe exposes the dataset classes through the public ``acoupipe.datasets`` package façade.
For dataset-specific APIs, the canonical package modules are ``acoupipe.datasets.synthetic``,
``acoupipe.datasets.miracle``, ``acoupipe.datasets.sriracha``, and ``acoupipe.datasets.ism``.
The former ``acoupipe.datasets.experimental`` module is no longer part of the package layout.

.. list-table:: Public dataset imports
    :header-rows: 1
    :widths: 24 38 38

    *   - Dataset
        - Recommended import
        - Canonical API reference
    *   - ``DatasetSynthetic``
        - ``from acoupipe.datasets.synthetic import DatasetSynthetic``
        - :class:`~acoupipe.datasets.synthetic.dataset.DatasetSynthetic`
    *   - ``DatasetMIRACLE``
        - ``from acoupipe.datasets import DatasetMIRACLE``
        - :class:`~acoupipe.datasets.miracle.dataset.DatasetMIRACLE`
    *   - ``DatasetSRIRACHA``
        - ``from acoupipe.datasets import DatasetSRIRACHA``
        - :class:`~acoupipe.datasets.sriracha.dataset.DatasetSRIRACHA`

.. toctree::
    :maxdepth: 1

    datasets/quickstart
    DatasetSynthetic <datasets/synthetic>
    DatasetMIRACLE <datasets/miracle>
    DatasetSRIRACHA <datasets/sriracha>
    datasets/features
    datasets/store
    jupyter/modify

