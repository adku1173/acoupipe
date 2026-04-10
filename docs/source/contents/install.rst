
Install
============

AcouPipe is currently available as a source-only package and is not published on PyPI or Conda. 

Tools
-----

There are different tools for Python environment management and package installation. Select your preferred method by clicking on one of the tabs below.

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        `"An extremely fast Python package and project manager, written in Rust."`

        This method only requires an installation of `uv <https://docs.astral.sh/uv/#installation>`_ itself. Choose this for a beginner-friendly and streamlined experience with faster dependency resolution.

    .. tab-item:: ``pip`` & ``venv``
        :sync: pip

        `"The PyPA recommended tool for installing Python packages."`

        This method requires Python and an installation of `pip <https://pip.pypa.io/en/stable/installation>`_. Choose this for a traditional Python experience.

Clone Repository
----------------

First, clone the AcouPipe repository and navigate to the directory:

.. code-block:: console

   $ git clone https://github.com/adku1173/acoupipe.git
   $ cd acoupipe

Virtual Environment
-------------------

We strongly encourage the use of virtual environments. An environment can be created with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv venv

        .. note::
            ``uv`` will handle environment activation implicitly (the environment is created at ``.venv``).

    .. tab-item:: ``venv``
        :sync: pip

        .. code-block:: console

            $ python3 -m venv my-env

        and activate the environment with:

        .. code-block:: console

            $ source my-env/bin/activate  # Linux/macOS
            $ my-env\Scripts\activate     # Windows

Installation
------------

Then, install AcouPipe with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv pip install -e .

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ pip install -e .

Dependencies
------------

AcouPipe depends on the following packages which will be installed automatically:

* Acoular_ (>=26.1)
* Ray_
* h5py_
* tqdm_
* pooch_

Dependency Groups
^^^^^^^^^^^^^^^^^

AcouPipe uses PEP 735 dependency groups for optional dependencies. The following groups are available:

* ``docs`` - Documentation building (Sphinx, IPython, matplotlib, etc.)
* ``lint`` - Code quality tools (ruff)
* ``test`` - Testing tools (pytest, pytest-cov, pytest-regtest, tensorflow, pandas)
* ``dev`` - All development dependencies (combines docs, lint, and test)
* ``unsupported`` - Experimental dependencies (pyroomacoustics)

To install with additional dependency groups:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv sync --group dev          # Install development dependencies
            $ uv sync --group test         # Install testing dependencies only
            $ uv sync --all-groups         # Install all dependency groups

    .. tab-item:: ``pip``
        :sync: pip

        .. note::
            Standard pip does not support PEP 735 dependency groups directly. Install specific packages manually:

        .. code-block:: console

            $ pip install -e .
            $ pip install pytest pytest-cov pytest-regtest tensorflow pandas  # test deps
            $ pip install sphinx matplotlib ipython sphinx-autoapi nbsphinx   # docs deps


Using a pre-build Docker image
------------------------------

If you are familiar with Docker_, the easiest way to use AcouPipe is by using an existing Docker image from DockerHub_. There are several images available, each tagged with the version of AcouPipe that is installed. The latest version is tagged as ``latest``.

The following images are available:

* ``adku1173/acoupipe:latest-base`` 
* ``adku1173/acoupipe:latest-full`` 
* ``adku1173/acoupipe:latest-dev`` 
* ``adku1173/acoupipe:latest-jupyter-gpu`` 

If  Docker_ is allready installed, simply pull the latest image with the command

.. code-block:: console

    $ docker pull adku1173/acoupipe:latest-full


