from acoular import PowerSpectraImport, SteeringVector
from acoular.internal import digest

import numpy as np
import scipy.linalg as spla
from numpy.random import default_rng
from traits.api import (
    CArray,
    CInt,
    Either,
    Float,
    Instance,
    Int,
    Map,
    Property,
    Trait,
    cached_property,
    property_depends_on,
)


class PowerSpectraAnalytic(PowerSpectraImport):
    #: equivalent degrees of freedom
    df_eq = Property()

    window = Map(
        {
            'Rectangular': np.ones,
            'Hanning': np.hanning,
            'Hamming': np.hamming,
            'Bartlett': np.bartlett,
            'Blackman': np.blackman,
        },
        default_value='Rectangular',
    )

    num_samples = CInt

    sample_freq = Float(1.0, desc='sampling frequency')

    #: Overlap factor for averaging: 'None'(default), '50%', '75%', '87.5%'.
    overlap = Trait('None', {'None': 1, '50%': 2, '75%': 4, '87.5%': 8}, desc='overlap of FFT blocks')

    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 65536,
    #: defaults to 1024.
    block_size = Trait(
        1024,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        desc='number of samples per FFT block',
    )

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(desc='overall number of FFT blocks')

    Q = CArray(shape=(None, None, None), dtype=complex, desc='source strengths matrix')

    noise = Either(CArray(shape=(None, None, None), dtype=complex), None, default=None, desc='noise covariance matrix')

    steer = Instance(SteeringVector)

    custom_transfer = Either(None, CArray, default=None)

    #: Mode of calculation of the cross spectral matrix. With 'analytic', the expected
    #: cross spectral matrix :math:`C = H Q H^H + N` is returned. With 'wishart', a sample
    #: cross spectral matrix is drawn from the complex Wishart distribution
    #: :math:`\alpha \hat{C} \sim \mathcal{CW}(\alpha, C)` with :math:`\alpha` = :attr:`df_eq`.
    #: If :attr:`noise` is None, the (nonsingular) source matrix :math:`Q` is used as the scale
    #: matrix instead, i.e. :math:`\hat{C} = H \hat{Q} H^H`.
    mode = Either('analytic', 'wishart', default='analytic', desc='mode of calculation of the cross spectral matrix')

    #: the state of the random variable. only relevant if :attr:`mode` is 'wishart'
    seed = Int(1, desc='seed value or random state of the random variable')

    #: The cross spectral matrix,
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property(desc='cross spectral matrix')

    frequencies = Property()

    def _get_frequencies(self):
        return self.fftfreq()

    # internal identifier
    digest = Property(
        depends_on=[
            'seed',
            'mode',
            'steer.digest',
            'Q',
            'noise',
            'ind_low',
            'ind_high',
            'sample_freq',
            'block_size',
            'overlap',
            'window',
            'num_samples',
            'custom_transfer',
        ],
    )

    @property_depends_on('num_samples, block_size, overlap, window')
    def _get_df_eq(self):
        w = self.window_(self.block_size)
        D = self.block_size if self.overlap == 'None' else int(self.block_size / self.overlap_)
        K = int(self.num_blocks)
        denom = (w @ w) ** 2
        fac = 0.0
        for j in range(1, K):
            shift = j * D
            num = w[:-shift] @ w[shift:]
            if num == 0.0:
                break
            rho = (num * num) / denom
            rho *= (K - j) / K
            fac += rho
        return K / (1 + 2 * fac)

    @cached_property
    def _get_digest(self):
        return digest(self)

    def fftfreq(self):
        """
        Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return abs(np.fft.fftfreq(self.block_size, 1.0 / self.sample_freq)[: int(self.block_size / 2 + 1)])

    @property_depends_on('num_samples, block_size, overlap')
    def _get_num_blocks(self):
        return self.overlap_ * self.num_samples / self.block_size - self.overlap_ + 1

    def _get_num_channels(self):
        return self.steer.mics.num_mics

    def _validate_custom_transfer(self):
        nfftfreq = self.fftfreq().shape[0]
        numsources = self.Q.shape[1]
        nummics = self.steer.mics.num_mics
        if self.custom_transfer.shape != (nfftfreq, nummics, numsources):
            msg = (
                'The shape of the custom transfer function must match the number of frequencies, microphones and sources!'
                f'but are {self.custom_transfer.shape} and {(nfftfreq, nummics, numsources)}'
            )
            raise ValueError(
                msg,
            )

    def _validate_freq_data(self):
        nfftfreq = self.fftfreq().shape[0]
        if nfftfreq != self.Q.shape[0]:
            msg = 'The number of frequencies must match the number of rows in the source strengths matrix!'
            raise ValueError(msg)
        if self.noise is not None and nfftfreq != self.noise.shape[0]:
            msg = 'The number of frequencies must match the number of rows in the noise matrix!'
            raise ValueError(msg)

    def _validate_df_eq(self, dim):
        if self.df_eq <= dim - 1:
            msg = (
                f'Wishart sampling requires the equivalent degrees of freedom (df_eq={self.df_eq:.2f}) '
                f'to exceed dim - 1 = {dim - 1}, where dim is the dimension of the scale matrix. '
                'Increase num_samples or decrease block_size.'
            )
            raise ValueError(msg)

    def _sample_wishart(self, scale, rng):
        r"""Draw a sample CSM with :math:`\alpha \hat{C} \sim \mathcal{CW}(\alpha, \Sigma)`.

        Uses Bartlett's decomposition :math:`W = T T^H \sim \mathcal{CW}(\alpha, I)` with a lower
        triangular :math:`T` and the linear transformation property with the Cholesky factor
        :math:`\Sigma = R R^H`, i.e. :math:`\hat{C} = \alpha^{-1} (R T)(R T)^H`.

        Parameters
        ----------
        scale : numpy.ndarray
            Hermitian positive definite scale matrix :math:`\Sigma` of shape (dim, dim).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Sample cross spectral matrix of shape (dim, dim).
        """
        df = self.df_eq
        dim = scale.shape[0]
        R = spla.cholesky(scale, lower=True)
        T = np.zeros((dim, dim), dtype=complex)
        # off-diagonal elements: standard complex normal, t_ij ~ CN(0, 1)
        n_tril = dim * (dim - 1) // 2
        T[np.tril_indices(dim, k=-1)] = (rng.normal(size=n_tril) + 1j * rng.normal(size=n_tril)) * np.sqrt(0.5)
        # diagonal elements: row i (0-based) of the lower triangular T has i off-diagonal entries,
        # hence t_ii^2 ~ Gamma(df - i, 1) such that E[W_ii] = df
        T[np.diag_indices(dim)] = np.sqrt(rng.gamma(df - np.arange(dim), scale=1.0))
        A = R @ T
        return A @ A.conj().T / df

    def _get_transfer(self):
        """Return the transfer matrix H of shape (num_freqs, num_mics, num_sources)."""
        if self.custom_transfer is not None:
            self._validate_custom_transfer()
            return self.custom_transfer
        fftfreq = self.fftfreq()
        H = np.zeros((fftfreq.shape[0], self.steer.mics.num_mics, self.Q.shape[1]), dtype=complex)
        for i in self.indices:  # calculate only the indices that are needed
            H[i] = self.steer.transfer(fftfreq[i]).T
        return H

    @property_depends_on(
        'seed, mode, block_size, overlap, window, sample_freq, num_samples, Q, noise, ind_low, ind_high, '
        'steer.digest, custom_transfer',
    )
    def _get_csm(self):
        H = self._get_transfer()
        if self.mode == 'analytic':
            csm = H @ self.Q @ H.swapaxes(2, 1).conjugate()
            if self.noise is not None:
                csm += self.noise
            return csm
        # mode == 'wishart'
        csm = np.zeros((H.shape[0], H.shape[1], H.shape[1]), dtype=complex)
        if self.noise is not None:
            # sample in microphone space with scale matrix C = H Q H^H + N
            # (includes the signal-noise cross terms)
            self._validate_df_eq(H.shape[1])
            for i in self.indices:
                rng = default_rng([self.seed, i])
                C = H[i] @ self.Q[i] @ H[i].conj().T + self.noise[i]
                csm[i] = self._sample_wishart(C, rng)
        else:
            # without noise, C = H Q H^H is singular (rank <= number of sources);
            # sampling in source space with scale matrix Q and transforming with H is equivalent
            self._validate_df_eq(self.Q.shape[1])
            for i in self.indices:
                rng = default_rng([self.seed, i])
                csm[i] = H[i] @ self._sample_wishart(self.Q[i], rng) @ H[i].conj().T
        return csm
