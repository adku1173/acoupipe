
import acoular as ac
import numpy as np
import numpy.fft as fft
from acoular.fastFuncs import calcTransfer
from acoular.internal import digest
from numpy import diag_indices, dot, r_, tril_indices, zeros
from numpy.random import default_rng
from scipy.linalg import cholesky
from traits.api import (
    Any,
    CArray,
    CLong,
    Either,
    Float,
    HasPrivateTraits,
    Instance,
    Int,
    Property,
    Trait,
    TraitError,
    cached_property,
    property_depends_on,
)

from acoupipe.config import GPURIR, PYROOMACOUSTICS
from acoupipe.datasets.utils import blockwise_transfer


class TransferBase(HasPrivateTraits):
    """Base class for transfer function implementations."""

    #: :class:`~acoular.environments.Environment` or derived object,
    #: which provides information about the sound propagation in the medium.
    #: Defaults to standard :class:`~acoular.environments.Environment` object (created lazily on
    #: first access).
    env = Instance(ac.Environment, args=())

    #: :class:`~acoular.grids.Grid`-derived object that provides the source locations.
    grid = Instance(ac.Grid, desc="beamforming grid")

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Instance(ac.MicGeom, desc="microphone geometry")

    #: Reference position or distance at which to evaluate the sound pressure
    #: of a grid point.
    #: If set to a scalar, this is used as reference distance to the grid points.
    #: If set to a vector, this is interpreted as x,y,z coordinates of the reference position.
    #: Defaults to [0.,0.,0.].
    ref = Property(desc="reference position or distance")

    # mirror trait for ref
    _ref = Any(np.array([0.0, 0.0, 0.0]), desc="reference position or distance")

    # internal identifier
    digest = Property(depends_on=["env.digest", "grid.digest", "mics.digest", "_ref"])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _set_ref(self, ref):
        if np.isscalar(ref):
            try:
                self._ref = np.absolute(float(ref))
            except ValueError as ve:
                raise TraitError(args=self, name="ref", info="Float or CArray(3,)", value=ref) from ve
        elif len(ref) == 3:
            self._ref = np.array(ref, dtype=float)
        else:
            raise TraitError(args=self, name="ref", info="Float or CArray(3,)", value=ref)

    def _get_ref(self):
        return self._ref

    def transfer(self, f, pos):
        pass


class TransferMonopole(TransferBase):

    # Sound travel distances from microphone array center to grid
    # points or reference position (readonly). Feature may change.
    r0 = Property(desc="array center to grid distances")

    # Sound travel distances from array microphones to grid
    # points (readonly). Feature may change.
    rm = Property(desc="all array mics to grid distances")

    @property_depends_on("grid.digest, env.digest, _ref")
    def _get_r0(self):
        if np.isscalar(self.ref):
            if self.ref > 0:
                return np.full((self.grid.size,), self.ref)
            return self.env._r(self.grid.pos())
        return self.env._r(self.grid.pos(), self.ref[:, np.newaxis])

    @property_depends_on("grid.digest, mics.digest, env.digest")
    def _get_rm(self):
        return np.atleast_2d(self.env._r(self.grid.pos(), self.mics.mpos))

    def transfer(self, f, ind=None):
        """Calculate the transfer matrix for one frequency.

        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed

        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency

        """
        if ind is None:
            trans = calcTransfer(self.r0, self.rm, np.array(2 * np.pi * f / self.env.c))
        elif not isinstance(ind, np.ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][np.newaxis], np.array(2 * np.pi * f / self.env.c))  # [0, :]
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], np.array(2 * np.pi * f / self.env.c))
        return trans


class TransferCustom(TransferBase):

    #: Transfer function data to be imported. Must be a complex array of shape (nfreqs, ngridpts, nmics).
    data = CArray(dtype=complex, shape=(None, None, None), desc="imported transfer function data")

    #: Frequencies of the transfer function data. Must match the first dimension of the data array.
    freqs = CArray(dtype=float, shape=(None,), desc="frequencies of the transfer function data")

    # internal identifier
    digest = Property(depends_on=["env.digest", "grid.digest", "mics.digest", "_ref", "freqs", "data"])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _validate(self, f):
        if self.freqs.shape[0] != self.data.shape[0]:
            msg = "Number of frequencies must match the first dimension of the data array."
            raise ValueError(msg)
        if f not in self.freqs:
            msg = f"Frequency {f} not found in the transfer function data."
            raise ValueError(msg)

    def transfer(self, f, ind=None):
        """Get the transfer matrix for one frequency.

        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed

        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency

        """
        self._validate(f)
        find = np.where(self.freqs == f)[0]
        if ind is None:
            return self.data[find]
        return np.atleast_2d(self.data[find, ind])



class PowerSpectraAnalytic(ac.PowerSpectraImport):

    transfer = Instance(TransferBase)

    numsamples = CLong

    sample_freq = Float(1.0,
        desc="sampling frequency")

    #: Overlap factor for averaging: 'None'(default), '50%', '75%', '87.5%'.
    overlap = Trait("None", {"None":1, "50%":2, "75%":4, "87.5%":8},
        desc="overlap of FFT blocks")

    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 65536,
    #: defaults to 1024.
    block_size = Trait(1024, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        desc="number of samples per FFT block")

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(
        desc="overall number of FFT blocks")

    Q = CArray(shape=(None,None,None), dtype=complex,
        desc="source strengths matrix")

    noise = Either(CArray(shape=(None,None,None), dtype=complex), None, default=None,
        desc="noise covariance matrix")

    mode = Either("analytic", "wishart", default="analytic",
        desc="mode of calculation of the cross spectral matrix")

    #: the state of the random variable. only relevant if :attr:`mode` is 'wishart'
    seed = Int(1,
        desc="seed value or random state of the random variable")

    #: The cross spectral matrix,
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property(
        desc="cross spectral matrix")

    frequencies = Property()

    _Q = Property()
    _noise = Property()


    def _get_frequencies(self):
        return self.fftfreq()

    # internal identifier
    digest = Property(
        depends_on = ["_csmsum", "seed", "mode","transfer.digest","Q","noise","ind_low","ind_high", "sample_freq",\
            "block_size", "overlap", "numsamples"],
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return abs(fft.fftfreq(self.block_size, 1./self.sample_freq)\
                        [:int(self.block_size/2+1)])

    @property_depends_on("numsamples, block_size, overlap")
    def _get_num_blocks ( self ):
        return self.overlap_*self.numsamples/self.block_size-\
        self.overlap_+1

    def _validate_freq_data ( self ):
        nfftfreq = self.fftfreq().shape[0]
        if nfftfreq != self.Q.shape[0]:
            raise ValueError(
                "The number of frequencies must match the number of rows in the source strengths matrix!")
            if self.noise is not None:
                if nfftfreq != self.noise.shape[0]:
                    raise ValueError(
                        "The number of frequencies must match the number of rows in the noise matrix!")

    def _sample_wishart ( self, scale, rng ):
        df = int(self.num_blocks)
        dim = scale.shape[0]
        if df <= dim:
            raise ValueError(f"Degrees of freedom ({df}) must be greater than the dimension of the scale matrix ({dim})")
        n_tril = dim * (dim-1) // 2
        C = cholesky(scale, lower=True)
        covariances = rng.normal(size=n_tril) + 1j*rng.normal(size=n_tril)
        # diagonal elements follow random gamma distribution (according to Nagar and Gupta, 2011)
        variances = (r_[[rng.gamma(df-dim +i, scale=1,size=1)**0.5
                        for i in range(dim)]])
        A = zeros(C.shape,dtype=complex)
        # input the covariances
        tril_idx = tril_indices(dim, k=-1)
        A[tril_idx] = covariances
        # Input the variances
        A[diag_indices(dim)] = variances.astype(complex)[:,0]
        # build matrix
        CA = dot(C, A)
        return dot(CA, CA.conjugate().T)/df

    @property_depends_on(
        "seed, mode, block_size, overlap, sample_freq, numsamples, Q, noise, ind_low, ind_high, transfer.digest")
    def _get_csm ( self ):
        return self._calc_csm()

    def _calc_csm( self ):
        Q = self._Q
        fftfreq = self.fftfreq()
        H = zeros((fftfreq.shape[0],self.transfer.mics.num_mics,Q.shape[1]),dtype=complex)
        for i in self.indices: # calculate only the indices that are needed
            H[i] = self.transfer.transfer(fftfreq[i]).T # transfer functions
        csm = H@Q@H.swapaxes(2,1).conjugate()
        if self.noise is not None:
            csm += self._noise
        return csm

    @property_depends_on("seed, mode, block_size, overlap, sample_freq, numsamples, Q, ind_low, ind_high")
    def _get__Q(self):
        if self.mode == "analytic":
            return self.Q
        else:
            fftfreq = self.fftfreq()
            Q = zeros((fftfreq.shape[0],self.Q.shape[1],self.Q.shape[1]),dtype=complex)
            for i in self.indices:
                rng = default_rng([self.seed,i])
                Q[i] = self._sample_wishart(self.Q[i],rng)
            return Q

    @property_depends_on("seed, mode, block_size, overlap, sample_freq, numsamples, noise, ind_low, ind_high")
    def _get__noise(self):
        if self.noise is not None:
            if self.mode == "analytic":
                return self.noise
            else:
                fftfreq = self.fftfreq()
                noise = zeros((fftfreq.shape[0],self.noise.shape[1],self.noise.shape[1]),dtype=complex)
                for i in self.indices:
                    rng = default_rng([self.seed+1,i])
                    noise[i] = self._sample_wishart(self.noise[i],rng)
                return noise



class TransferISM(TransferBase):

    sample_freq = Float(1.0,
        desc="sampling frequency")

    block_size = Int(1024,
        desc="number of samples per FFT block")

    room_size = CArray(shape=(3,), dtype=float, desc="room size")

    alpha = CArray(shape=(6,), desc="Sabine absorption coefficients for each wall")

    beta = Property(desc="reflection coefficients")

    order = Int(desc="number of images to simulate")

    rir = Property()

    rel_tdir = CArray(shape=(3,), dtype=float, # TODO: can also be rotated
        desc="relative translation of the origin inside the vaild area of the room (values between 0 and 1)")

    origin = Property()

    _transfer = Property(depends_on=["digest"])

    def _get_origin(self):
        if np.isscalar(self.ref):
            all_locs = np.concatenate([self.grid.gpos, self.mics.mpos], axis=1)
        else:
            all_locs = np.concatenate([self.grid.gpos, self.mics.mpos, self.ref[:, np.newaxis]], axis=1)
        rmax = np.array([w.corners.max(1) for w in self.room.walls]).max(0)
        return self.rel_tdir*(rmax - np.max(all_locs,axis=1) - np.min(all_locs,axis=1))

    def _get_beta(self):
        return np.sqrt(1 - self.alpha)

    def sabine(self):
        V = np.prod(self.room_size)
        A = (self.alpha[0]+self.alpha[1]) * self.room_size[1]*self.room_size[2] + \
            (self.alpha[2]+self.alpha[3]) * self.room_size[0]*self.room_size[2] + \
            (self.alpha[4]+self.alpha[5]) * self.room_size[0]*self.room_size[1]
        return 0.1611 * V / A

    @cached_property
    def _get__transfer(self):
        init_shape = self.rir.shape
        tf = blockwise_transfer(self.rir.reshape(-1, init_shape[-1]), self.block_size)
        return tf.reshape((*init_shape[:-1], tf.shape[-1]))

    def transfer(self, f, ind=None):
        """Get the transfer matrix for one frequency.

        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed

        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency

        """
        freqs = abs(fft.fftfreq(self.block_size, 1.0 / self.sample_freq)[: int(self.block_size / 2 + 1)])
        find = np.where(freqs == f)[0]
        if ind is None:
            return self._transfer[find]
        return np.atleast_2d(self._transfer[find, ind])


if GPURIR:
    import gpuRIR
    from traits.api import Enum, List

    class TransferShoeBoxGPUrir(TransferISM):

        order = Property()

        _order = Either((
            CArray(shape=(3,), dtype=float), None), default=None,
            desc="number of images to simulate")

        att_diff = Either(
            (Float, None),
            default=30,
            desc="Attenuation when using diffuse reverberation model (in dB)")

        att_max = Either(
            (Float, None),
            default=60,
            desc="Maximum attenuation of the room (in dB)")

        source_directivity = Either(
            (
            Enum("omni", "homni", "card", "hypcard", "subcard", "bidir"),
            List(Enum("omni", "homni", "card", "hypcard", "subcard", "bidir"))),
            default="omni",
            )

        source_orientation = Either((None, CArray(dtype=float)), default=None)

        mics_directivity = Either(
            (
            Enum("omni", "homni", "card", "hypcard", "subcard", "bidir"),
            List(Enum("omni", "homni", "card", "hypcard", "subcard", "bidir"))),
            default="omni",
            )

        mics_orientation = Either((None, CArray(dtype=float)), default=None)

        _tdiff = Property()

        def get__tdiff(self):
            if self.att_diff is not None:
                return gpuRIR.att2t_SabineEstimator(self.att_diff, self.sabine())
            return None

        def _get_order(self):
            if self._order is not None:
                return self._order

            return gpuRIR.t2n(self.Tdiff, self.room_size)

        def _set_order(self, order):
            self._order = order

        def _get_rir(self):
            tmax= self.att_max/60.0 * self.sabine()
            rir = np.zeros((self.grid.size, self.mics.num_mics, int(self.sample_freq*tmax)))
            kwargs = {
                "room_sz" : self.room.size,
                "beta" : self.beta,
                "nb_img" : self.order,
                "Tmax" : tmax,
                "fs" : self.sample_freq,
                "Tdiff" : self._tdiff,
                "c" : self.env.c,

            }

            if isinstance(self.source_directivity, str) and isinstance(self.mics_directivity, str) and \
                (self.source_orientation is None or self.source_orientation.shape == (3,)) and \
                    (self.mics_orientation is None or self.mics_orientation.shape == (3,)):

                # calculate at once
                rir[:,:,:] = gpuRIR.simulateRIR(
                    pos_src = self.grid.gpos.T + self.origin[np.newaxis, :],
                    pos_rcv = self.mics.mpos.T + self.origin[np.newaxis, :],
                    spkr_pattern = self.source_directivity,
                    mic_pattern = self.mics_directivity,
                    orV_src = self.source_orientation,
                    orV_rcv = self.mics_orientation,
                    **kwargs
                    )
            elif (isinstance(self.source_directivity, list) and isinstance(self.mics_directivity, list)) or \
                ((self.source_orientation is not None and self.source_orientation.size > 3) and \
                    (self.mics_orientation is not None or self.mics_orientation.size > 3)):
                # calculate one by one
                if not isinstance(self.source_directivity, list):
                    self.source_directivity = [self.source_directivity] * self.grid.size
                if not isinstance(self.mics_directivity, list):
                    self.mics_directivity = [self.mics_directivity] * self.mics.num_mics
                if self.source_orientation is not None and self.source_orientation.size > 3:
                    self.source_orientation = np.tile(self.source_orientation, (self.grid.size, 1))
                if self.mics_orientation is not None and self.mics_orientation.size > 3:
                    self.mics_orientation = np.tile(self.mics_orientation, (self.mics.num_mics, 1))
                for i in range(self.grid.size):
                    for j in range(self.mics.nummics):
                        rir[i, j] = gpuRIR.simulateRIR(
                            pos_src = self.grid.gpos.T[i] + self.origin,
                            pos_rcv = self.mics.mpos.T[j] + self.origin,
                            spkr_pattern = self.source_directivity[i],
                            mic_pattern = self.mics_directivity[j],
                            orV_src = self.source_orientation[i],
                            orV_rcv = self.mics_orientation[j],
                            **kwargs
                        )
            elif isinstance(self.source_directivity, list) or (
                self.source_orientation is not None and self.source_orientation.size > 3):
                if not isinstance(self.source_directivity, list):
                    self.source_directivity = [self.source_directivity] * self.grid.size
                if self.source_orientation is not None and self.source_orientation.shape == (3,):
                    self.source_orientation = np.tile(self.source_orientation, (self.grid.size, 1))

                for i in range(self.grid.size):
                    rir[i,:,:] = gpuRIR.simulateRIR(
                        pos_src = self.grid.gpos.T[i] + self.origin,
                        pos_rcv = self.mics.mpos.T + self.origin,
                        spkr_pattern = self.source_directivity[i],
                        mic_pattern = self.mics_directivity,
                        orV_src = self.source_orientation[i],
                        orV_rcv = self.mics_orientation,
                        **kwargs
                    )

            elif isinstance(self.mics_directivity, list) or (
                self.mics_orientation is not None and self.mics_orientation.size > 3):
                if not isinstance(self.mics_directivity, list):
                    self.mics_directivity = [self.mics_directivity] * self.mics.num_mics
                if self.mics_orientation is not None and self.mics_orientation.shape == (3,):
                    self.mics_orientation = np.tile(self.mics_orientation, (self.mics.num_mics, 1))

                for j in range(self.mics.num_mics):
                    rir[:,j] = gpuRIR.simulateRIR(
                        pos_src = self.grid.gpos.T + self.origin,
                        pos_rcv = self.mics.mpos.T[j] + self.origin,
                        spkr_pattern = self.source_directivity,
                        mic_pattern = self.mics_directivity[j],
                        orV_src = self.source_orientation,
                        orV_rcv = self.mics_orientation[j],
                        **kwargs
                    )
            return rir


if PYROOMACOUSTICS:
    import pyroomacoustics as pra
    from traits.api import Bool, List

    class TransferShoeBoxPyroomacoustics(TransferISM):

        room = Instance(pra.ShoeBox, desc="room geometry")

        rir = Property(depends_on=["translation", "room", "mics.digest","grid.digest","env.digest", "pad"], desc="room impulse response")

        digest = Property(depends_on=["translation", "room","env.digest", "grid.digest", "mics.digest", "pad","_ref"])

        pad = Bool(False, desc="pad RIRs with zeros to the length of the longest one. Otherwise, trim all RIRs to the length of the shortest one.")

        source_directivity = List()

        receiver_directivity = List()

        def transfer(self, f, ind=None):
            if ind is None:
                return self.room.transfer(f, self.grid.pos())
            return self.room.transfer(f, self.grid.pos()[ind])

        @cached_property
        def _get_rir(self):
            """Compute the RIRs for the given room, source and microphone locations."""
            origin = self.origin
            self.room.sources = [pra.SoundSource(loc+origin) for loc in self.grid.gpos.T] # TODO: add directivity
            self.room.add_microphone_array(
                pra.MicrophoneArray(self.mics.mpos+origin[:,np.newaxis], fs=self.room.fs))

            self.room.compute_rir() # self.room.rir is a list of lists of RIRs with different length
            n = len(self.room.rir[0][0]) # length of the first rir
            nsrc = self.grid.size
            nmics = self.mics.num_mics

            if not self.pad:
                # trim all RIRs by the length of the shortest one
                rir_arr = np.zeros((nsrc, nmics,n))
                for j in range(nsrc):
                    for i in range(nmics):
                        rir = np.array(self.room.rir[i][j])
                        ns = min(n, rir.shape[0])
                        rir_arr[j,i,:ns] = rir[:ns]
                return rir_arr[:,:,:ns]
            else:
                for j in range(nsrc):
                    for i in range(nmics):
                        n = max(n, np.array(self.room.rir[i][j]).shape[0])
                rir_arr = np.zeros((nsrc, nmics,n))
                for j in range(nsrc):
                    for i in range(nmics):
                        rir = np.array(self.room.rir[i][j])
                        ns = rir.shape[0]
                        rir_arr[j,i,:ns] = rir
            # trim the RIRs to the a power of 2 length
            lpow2 = 2**int(np.ceil(np.log2(rir.shape[2])))
            padded_rir = np.zeros((rir.shape[0], rir.shape[1], lpow2))
            padded_rir[:,:,:rir.shape[2]] = rir
            return rir_arr
