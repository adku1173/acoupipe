import numpy.fft as fft
from acoular import PowerSpectraImport, SteeringVector
from acoular.internal import digest
from numpy import diag_indices, dot, r_, tril_indices, zeros
from numpy.random import default_rng
from scipy.linalg import cholesky
from traits.api import CArray, CInt, Either, Float, Instance, Int, Property, Trait, cached_property, property_depends_on


class PowerSpectraAnalytic(PowerSpectraImport):
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

    _Q = Property()

    _noise = Property()

    Q = CArray(shape=(None, None, None), dtype=complex, desc='source strengths matrix')

    noise = Either(CArray(shape=(None, None, None), dtype=complex), None, default=None, desc='noise covariance matrix')

    steer = Instance(SteeringVector)

    custom_transfer = Either(None, CArray, default=None)

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
            '_csmsum',
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
            'num_samples',
        ],
    )

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
        return abs(fft.fftfreq(self.block_size, 1.0 / self.sample_freq)[: int(self.block_size / 2 + 1)])

    @property_depends_on('num_samples, block_size, overlap')
    def _get_num_blocks(self):
        return self.overlap_ * self.num_samples / self.block_size - self.overlap_ + 1

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

    def _sample_wishart(self, scale, rng):
        df = int(self.num_blocks)
        dim = scale.shape[0]
        n_tril = dim * (dim - 1) // 2
        C = cholesky(scale, lower=True)
        covariances = rng.normal(size=n_tril) + 1j * rng.normal(size=n_tril)
        # diagonal elements follow random gamma distribution (according to Nagar and Gupta, 2011)
        variances = r_[[rng.gamma(df - dim + i, scale=1, size=1) ** 0.5 for i in range(dim)]]
        A = zeros(C.shape, dtype=complex)
        # input the covariances
        tril_idx = tril_indices(dim, k=-1)
        A[tril_idx] = covariances
        # Input the variances
        A[diag_indices(dim)] = variances.astype(complex)[:, 0]
        # build matrix
        CA = dot(C, A)
        return dot(CA, CA.conjugate().T) / df

    @property_depends_on(
        'seed, mode, block_size, overlap, sample_freq, num_samples, Q, noise, ind_low, ind_high, steer.digest, custom_transfer',
    )
    def _get_csm(self):
        # self._validate_freq_data()
        if self.custom_transfer is None:
            return self._calc_csm()
        self._validate_custom_transfer()
        return self._calc_csm_custom()

    def _calc_csm(self):
        Q = self._Q
        fftfreq = self.fftfreq()
        H = zeros((fftfreq.shape[0], self.steer.mics.num_mics, Q.shape[1]), dtype=complex)
        for i in self.indices:  # calculate only the indices that are needed
            H[i] = self.steer.transfer(fftfreq[i]).T  # transfer functions
        csm = H @ Q @ H.swapaxes(2, 1).conjugate()
        if self.noise is not None:
            csm += self._noise
        return csm

    def _calc_csm_custom(self):
        Q = self._Q
        H = self.custom_transfer
        csm = H @ Q @ H.swapaxes(2, 1).conjugate()
        if self.noise is not None:
            csm += self._noise
        return csm

    @property_depends_on('seed, mode, block_size, overlap, sample_freq, num_samples, Q, ind_low, ind_high')
    def _get__Q(self):
        if self.mode == 'analytic':
            return self.Q
        fftfreq = self.fftfreq()
        Q = zeros((fftfreq.shape[0], self.Q.shape[1], self.Q.shape[1]), dtype=complex)
        for i in self.indices:
            rng = default_rng([self.seed, i])
            Q[i] = self._sample_wishart(self.Q[i], rng)
        return Q

    @property_depends_on('seed, mode, block_size, overlap, sample_freq, num_samples, noise, ind_low, ind_high')
    def _get__noise(self):
        if self.noise is not None:
            if self.mode == 'analytic':
                return self.noise
            fftfreq = self.fftfreq()
            noise = zeros((fftfreq.shape[0], self.noise.shape[1], self.noise.shape[1]), dtype=complex)
            for i in self.indices:
                rng = default_rng([self.seed + 1, i])
                noise[i] = self._sample_wishart(self.noise[i], rng)
            return noise
        return None
