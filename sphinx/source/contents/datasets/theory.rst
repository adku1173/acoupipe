.. _theory-section:

Theory
======

Notation
--------

* Boldface type for vectors and matrices, e.g., :math:`\mathbf{p}` represents a vector, while :math:`\mathbf{H}` denotes a matrix
* Calligraphic type for sets, e.g., :math:`\mathcal{G}` represents a set of ground-truth source characteristics
* Blackboard Bold is used to indicate number spaces, e.g., :math:`\mathbb{C}` represents the complex number space 

Problem and Concept
--------------------

In source characterization tasks, the goal is to approximate a set :math:`\mathcal{G}=\{\mathbf{y}_{j} \mid j=1, \ldots, J\}` of ground-truth source characteristics :math:`\mathbf{y}_j`. 
Often, this involves estimating the source location :math:`\mathbf{r}_j` and the squared sound pressure amplitude :math:`p^2_j(\mathbf{r}_0,\omega)` with respect to a reference position :math:`\mathbf{r}_0`, known as the source strength.
A common practice is to set the reference position :math:`\mathbf{r}_0` to a sensor location. 
In the AcouPipe datasets, the channel closest to the origin of the coordinate system is used. 

Propagation Model
------------------

Given :math:`M` spatially distributed receivers, :math:`J` uncorrelated and spatially stationary sources, and a linear propagation model, the complex sound pressure at the :math:`m`-th sensor is described by:

.. math::

   p(\mathbf{r}_{m}, \omega) = \sum_{j=1}^J h_{mj}(\omega) q(\mathbf{r}_{j}, \omega) + n(\boldsymbol{r}_{m}, \omega)

Here, :math:`\omega` is the angular frequency, :math:`h_{mj}` is the transfer function, and :math:`q(\mathbf{r}_{j}, \omega)` represents the complex-valued amplitude of the source. Independent noise is modeled as :math:`n(\boldsymbol{r}_{m}, \omega)`.

The propagation equation can also be written in matrix form:

.. math::

   \mathbf{p} = \mathbf{H}\mathbf{q} + \mathbf{n}

with :math:`\mathbf{p} \in \mathbb{C}^{M}`, :math:`\mathbf{q} \in \mathbb{C}^{J}`, :math:`\mathbf{n} \in \mathbb{C}^{M}`, and :math:`\mathbf{H} \in \mathbb{C}^{M\times J}`.

.. Source Strength and Reference Position
.. --------------------------------------

.. A common practice is to set the reference position :math:`\mathbf{r}_0` to a sensor location. 
.. In the AcouPipe datasets, the channel closest to the origin of the coordinate system is used. 
.. Then, the individual auto-power :math:`a_j(\mathbf{r}_0,\omega)` of the reference sensor induced by the :math:`j`-th source can be calculated:

.. .. math::

..    a_j(\mathbf{r}_0,\omega) = \frac{1}{B} \sum_{b=1}^{B} p_j(\mathbf{r}_0,\omega) p_j(\mathbf{r}_0,\omega)^*.

.. This serves as a measure of the individual source strength.

Cross-spectral Matrix (CSM)
---------------------------

The cross-spectral matrix (CSM) plays a vital role in various microphone array methods and is one of the features that can be generated with AcouPipe when the features attribute is set to :code:`features=['csm']` or :code:`features=['csmtriu']`. 
It contains the auto-power and cross-power spectra of all sensors. The CSM is a complex hermitian matrix and contains redundant information. 
By using :code:`features=['csmtriu']`, only the upper triangular part of the CSM is returned (the conjugate complex of the CSM is neglected; see :cite:`Castellini2021`). 

**analytical CSM (used in Dataset 2)**

If the matrix :math:`\mathbf{Q} \in \mathbb{C}^{J \times J}` containing the sources' auto- and cross-power spectra and the transfer matrix :math:`\mathbf{H} \in \mathbb{C}^{M \times J}` are known, the CSM can be calculated analytically as:

.. math::

   \mathbf{C} = \mathbb{E}[\mathbf{p}\mathbf{p}^{\text{H}}] = \mathbf{H} \mathbb{E}[ \mathbf{Q} ] \mathbf{H}^{\text{H}}

where :math:`\mathbb{E}[\cdot]` denotes the expectation operator. This enables a fast calculation of the CSM but neglects uncertainties that stem from a limited number of snapshots.
It can be used to generate the CSM for Dataset 2 when the attribute :code:`whishart` is set to :code:`False`.

**estimated CSM (used in Dataset1)**

In practice, the CSM is estimated from a finite number of samples. One common method for estimating the CSM is Welch's method:

.. math::

   \hat{\mathbf{C}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{p} \mathbf{p}^{\text{H}}

This method is the standard for Dataset 1 in order to obtain the CSM and the source strength labels. In Dataset 1, the time data
of each microphone is simulated for several seconds. A drawback is the higher computational cost of this method.

**sampled CSM (used in Dataset 2)**


By assuming stationary sources with non-deterministic source signals, a snapshot deficient CSM can be sampled. 
Given the matrix :math:`\mathbf{Q}`, it is possible to approximate :math:`\mathbf{Q}` using the Cholesky decomposition :math:`\mathbf{Q}(\omega) = \mathbf{U}\mathbf{U}^{\mathsf{H}}` and the Bartlett decomposition:

.. math::

   \hat{\mathbf{Q}}  = \frac{1}{n} \mathbf{U} \mathbf{A} \mathbf{U}^{\mathsf{H}}

Here, :math:`\mathbf{A}` is generated for :math:`n` different degrees of freedom, representing the number of snapshots. The distribution of :math:`\mathbf{A}` follows a complex Wishart distribution :math:`\mathcal{W}_{\mathbb{C}} (n,\mathrm{I})`.

Sampling the cross-spectral matrix is then achieved by multiplying the Wishart-distributed source matrix with the transfer matrix :math:`\mathbf{H}`:

.. math::

   \hat{\mathbf{C}}_{\mathcal{W}} = \mathbf{H} \hat{\mathbf{Q}} \mathbf{H}^{\mathsf{H}}.

This method can be used to generate the CSM for Dataset 2 when the attribute :code:`whishart` is set to :code:`True`.


CSM Eigenmodes 
--------------

The Eigenmodes of the CSM is a further feature that can be generated with AcouPipe when the features attribute is set to :code:`features=['eigmode']`.
The Eigenmodes of the CSM are the eigenvectors scaled by their corresponding eigenvalues and have been used in :cite:`Kujawski2022`.
Eigen-decomposition is used to decompose the CSM into its eigenvalues and eigenvectors:

.. math::

   \hat{\mathbf{C}} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{\text{H}}

Here, :math:`\mathbf{V}` contains the complex eigenvectors, and :math:`\mathbf{\Lambda}` is a diagonal matrix of eigenvalues. 


Conventional Beamforming 
------------------------

The conventional beamforming map is calculated by processing the CSM with the corresponding steering vector :math:`h`, such that  

.. math::

   b(\mathbf{x}_t) = \mathbf{h}^{\mathrm{H}}(\mathbf{x}_t) \mathbf{C h}(\mathbf{x}_t), \quad t \in \{1, \ldots, G\}.

The equation is evaluated for a spatial grid.


The conventional beamforming map is a feature with AcouPipe when the features attribute is set to :code:`features=['sourcemap']`.
For convenience, the sound radiation is assumed to come from a monopole. 
Different steering vector formulations exist in the literature, varying in terms of spatial precision and accuracy in determining the source strength. 
Formulation III according to :cite:`Sarradj2012` is used as the default, which is defined as:

.. math::

   h_m = \frac{1}{r_{t, 0} r_{t, m} \sum_{l=1}^M r_{t, l}^{-2}} \exp^{-\jmath k\left(r_{t, m}-r_{t, 0}\right)}

Here, :math:`r_{t, m}` refers to the distance between the steered location and the respective :math:`m`-th sensor, while :math:`r_{t, 0}` specifies the distance from the focus point to the reference point where the sound pressure is evaluated.
Sarradj demonstrated that using formulation III, the maximum sound pressure level depicted in a sound map may not precisely correspond to the true position of a single sound source. 
However, the study also revealed that the maximum does equal the true source strength for larger Helmholtz numbers.





