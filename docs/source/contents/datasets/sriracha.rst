.. _dataset_sriracha:

DatasetSRIRACHA
===============

``DatasetSRIRACHA`` is an experimental microphone array dataset generator that builds multi-source
scenes by convolving synthetic source signals with **measured** spatial room impulse responses (SRIRs)
from the `SRIRACHA`_ dataset :cite:`Pelling2025`. Multi-source scenes are realized by
superposition of individually convolved source signals.

SRIRACHA (Shoebox Room Impulse Response Archive with Varying Absorption) provides SRIRs measured in a
rectangular (“shoebox”) room with two absorption conditions. The dataset comprises eight scenarios that
combine two source–receiver plane distances, two source arrangements, and two room absorption settings.


Measurement environment and room configurations
-----------------------------------------------

The SRIRACHA measurements were conducted in a rigid concrete shoebox room with dimensions
:math:`6.22 \times 3.85 \times 3.07~\mathrm{m}` (length × width × height). Two room configurations are provided:

- **SR**: “empty” shoebox room configuration with foam absorbers placed in two corners to control low-frequency decay.
- **SRA**: shoebox room with additional absorption, realized by placing three absorber walls inside the room.

A simplified geometry of the absorbent configuration is provided with the dataset as an ``.stl`` file.

.. figure:: ../../_static/sriracha_sra.png
    :width: 750
    :align: center

    Experimental setup rendering for the absorbent room configuration (SRA). The planar microphone array is
    mounted in front of a motorized positioning system; absorber walls are shown in brown.


Reverberation time measurement
------------------------------

For both room configurations, reverberation times were measured according to DIN-EN-ISO 3382-2 using an
omnidirectional loudspeaker and omnidirectional microphone with an interrupted-noise procedure. The
measurement setup (SRA configuration) is shown below.

.. figure:: ../../_static/sriracha_t60-measurement.jpg
    :width: 750
    :align: center

    Photograph of the reverberation-time measurement setup in the SRA configuration (omnidirectional
    dodecahedron loudspeaker, microphone array, and absorber wall).


Scenarios
---------

SRIRACHA provides eight measurement scenarios. Naming follows:

- ``SR`` vs ``SRA``: room configuration (empty vs absorbent),
- ``1`` vs ``2``: source–receiver plane distance (near vs far),
- ``-D`` suffix: dense local source arrangement.

In AcouPipe, the scenario is selected via the ``scenario`` parameter of
:class:`acoupipe.datasets.experimental.DatasetSRIRACHA`.

.. list-table:: Available SRIRACHA scenarios (SRIR grids and meta-data)
    :header-rows: 1
    :widths: 10 22 12 10 12 12 12 20

    *   - Scenario
        - Room configuration
        - # sources
        - # mics
        - dx = dy
        - dz
        - Δx = Δy
        - Notes
    *   - SR1
        - Empty shoebox room (SR)
        - 64 × 64 = 4096
        - 64
        - 146.7 cm
        - 75.3 cm
        - 23.3 mm
        - Near distance, full grid
    *   - SRA1
        - Absorbent room (SRA)
        - 64 × 64 = 4096
        - 64
        - 146.7 cm
        - 75.3 cm
        - 23.3 mm
        - Near distance, full grid
    *   - SR1-D
        - Empty shoebox room (SR)
        - 33 × 33 = 1089
        - 64
        - 16.0 cm
        - 75.3 cm
        - 5.0 mm
        - Near distance, dense local grid
    *   - SRA1-D
        - Absorbent room (SRA)
        - 33 × 33 = 1089
        - 64
        - 16.0 cm
        - 75.3 cm
        - 5.0 mm
        - Near distance, dense local grid
    *   - SR2
        - Empty shoebox room (SR)
        - 64 × 64 = 4096
        - 64
        - 146.7 cm
        - 147.9 cm
        - 23.3 mm
        - Far distance, full grid
    *   - SRA2
        - Absorbent room (SRA)
        - 64 × 64 = 4096
        - 64
        - 146.7 cm
        - 148.0 cm
        - 23.3 mm
        - Far distance, full grid
    *   - SR2-D
        - Empty shoebox room (SR)
        - 33 × 33 = 1089
        - 64
        - 16.0 cm
        - 147.9 cm
        - 5.0 mm
        - Far distance, dense local grid
    *   - SRA2-D
        - Absorbent room (SRA)
        - 33 × 33 = 1089
        - 64
        - 16.0 cm
        - 148.0 cm
        - 5.0 mm
        - Far distance, dense local grid


Signal acquisition and post-processing
--------------------------------------

SRIRACHA SRIRs were acquired using a planar 64-channel microphone array (Vogel spiral, aperture 1.47 m)
and a motorized 2D positioning system to place an excitation loudspeaker on a source plane parallel to the array.
Excitation used an exponential sine sweep (3 s) covering 100 Hz to 16 kHz; recordings were 6 s long at 51.2 kHz
to capture the room decay. The processed signals are resampled to **32 kHz** and deconvolved to obtain impulse
responses. To reduce storage requirements, SRIRs are truncated based on a global decay criterion (instantaneous
average energy decay of at least 60 dB).

Environmental parameters (temperature and humidity) are recorded and used to compute the speed of sound.
Source locations are provided as both nominal and corrected coordinates.


Data format and coordinate systems
----------------------------------

The dataset is distributed as HDF5 files. Conceptually, each scenario contains an impulse-response tensor of
shape ``(n_sources, n_mics, n_samples)`` plus geometry and meta-data:

- ``/data/impulse_response``: float32 SRIR array
- ``/location/receiver``: float64 microphone coordinates (shape ``(n_mics, 3)``)
- ``/location/source``: float64 corrected source coordinates (shape ``(n_sources, 3)``)
- ``/location/source_raw``: float32 nominal source coordinates (shape ``(n_sources, 3)``)
- ``/metadata``: speed of sound, temperature, humidity, sampling rate, etc.

.. note::

    Large scenarios (e.g., ``SR1``, ``SR2``, ``SRA1``, ``SRA2``) may be split into multiple HDF5 chunks
    (e.g., ``SR1-C1.h5`` ... ``SR1-C4.h5``) to keep file sizes manageable.

The measurement coordinates are provided in a **left-handed** coordinate system for compatibility with MIRACLE.
The accompanying ``.stl`` geometry is defined in a **right-handed** coordinate system; when using the ``.stl`` as
reference, mirror the x-axis of the provided measurement coordinates.


Default FFT parameters
----------------------

The underlying default FFT parameters are:

.. table:: FFT Parameters

    ===================== ========================================
    Sampling Rate         fs = 32,000 Hz
    Block size            256 samples
    Block overlap         50 %
    Windowing             von Hann / Hanning
    ===================== ========================================


Randomized properties
---------------------

Several properties are randomized for each generated source case when building multi-source scenes from
SRIRACHA. By default, source positions are sampled from the discrete SRIR grid of the selected scenario.

.. table:: Randomized properties (defaults)

    ==================================================================   ===================================================
    No. of sources                                                       Poisson distributed (:math:`\\lambda = 3`)
    Source positions                                                     Bivariate normal distributed (:math:`\\sigma = 0.1688 d_a`)
    Source strength (:math:`[{Pa}^2]` at reference position)              Rayleigh distributed (:math:`\\sigma_R = 5`)
    Relative noise variance                                              Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
    ==================================================================   ===================================================


Example
-------

.. code-block:: python

    from acoupipe.datasets.experimental import DatasetSRIRACHA

    ds = DatasetSRIRACHA(scenario="SRA2-D")



License and access
------------------

SRIRACHA is distributed under **CC BY-NC-SA 4.0** (non-commercial). Please ensure your intended use complies
with the license terms.
