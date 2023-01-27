
from acoular import PowerSpectraImport, SteeringVector
from numpy import diag_indices, dot, r_, tril_indices, zeros
from numpy.random import RandomState
from scipy.linalg import cholesky
from traits.api import CArray, Either, Instance, Int, Property, property_depends_on


class PowerSpectraAnalytic(PowerSpectraImport):

    Q = CArray(shape=(None,None,None), dtype=complex, 
        desc="source strengths matrix")

    noise = Either(CArray(shape=(None,None,None), dtype=complex), None, default=None,
        desc="noise covariance matrix")
    
    steer = Instance(SteeringVector)

    mode = Either("analytic", "wishart", default="analytic",
        desc="mode of calculation of the cross spectral matrix")

    #: the state of the random variable. only relevant if :attr:`mode` is 'wishart'
    seed = Int(1,
        desc="random state of the random variable")

    df = Int(64,
        desc="degrees of freedom of the wishart distribution")

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")

    _Q_wishart = CArray(shape=(None,None,None), dtype=complex, 
        desc="source strengths matrix sampled from the wishart distribution")

    _noise_wishart = CArray(shape=(None,None,None), dtype=complex, 
        desc="noise covariance matrix sampled from the wishart distribution")

    def _validate_freq_data ( self ):
        if self.frequencies is None:
            raise ValueError(
                "The frequencies must be given but are None type!")
        else:
            if self.fftfreq().shape[0] != self.Q.shape[0]:
                raise ValueError(
                    "The number of frequencies must match the number of rows in the source strengths matrix!")

    def _sample_wishart ( self, scale, rng ):
        dim = scale.shape[0]
        n_tril = dim * (dim-1) // 2
        C = cholesky(scale, lower=True)
        covariances = rng.normal(size=n_tril) + 1j*rng.normal(size=n_tril)
        # diagonal elements follow random gamma distribution (according to Nagar and Gupta, 2011)
        variances = (r_[[rng.gamma(self.df-dim +i, scale=1,size=1)**0.5
                        for i in range(dim)]])
        A = zeros(C.shape,dtype=complex)
        # input the covariances
        tril_idx = tril_indices(dim, k=-1)
        A[tril_idx] = covariances
        # Input the variances
        A[diag_indices(dim)] = variances.astype(complex)[:,0]
        # build matrix
        CA = dot(C, A)
        return dot(CA, CA.conjugate().T)/self.df

    @property_depends_on("Q,noise,steer.digest,ind_low,ind_high")
    def _get_csm ( self ):
        self._validate_freq_data()
        if self.mode == "analytic":
            return self._calc_csm()
        elif self.mode == "wishart":
            return self._calc_csm_wishart()
        else:
            raise ValueError("The mode must be either 'analytic' or 'wishart'!")

    def _calc_csm( self ):
        fftfreq = self.fftfreq()
        H = zeros((fftfreq.shape[0],self.steer.mics.num_mics,self.Q.shape[1]),dtype=complex)
        for i in self.indices: # calculate only the indices that are needed
            H[i] = self.steer.transfer(fftfreq[i]).T # transfer functions
        csm = H@self.Q@H.swapaxes(2,1).conjugate()
        if self.noise is not None:
            csm += self.noise
        return csm        

    def _calc_csm_wishart( self ):
        rng1 = RandomState(self.seed)
        rng2 = RandomState(self.seed+1)
        fftfreq = self.fftfreq()
        H = zeros((fftfreq.shape[0],self.steer.mics.num_mics,self.Q.shape[1]),dtype=complex)
        self._Q_wishart = zeros((fftfreq.shape[0],self.Q.shape[1],self.Q.shape[1]),dtype=complex)
        if self.noise is not None:
            self._noise_wishart = zeros((fftfreq.shape[0],self.noise.shape[1],self.noise.shape[1]),dtype=complex)
        for i in self.indices:
            H[i] = self.steer.transfer(fftfreq[i]).T
            self._Q_wishart[i] = self._sample_wishart(self.Q[i],rng1)
            if self.noise is not None:
                self._noise_wishart[i] = self._sample_wishart(self.noise[i],rng2)
        csm = H@self._Q_wishart@H.swapaxes(2,1).conjugate()
        if self.noise is not None:
            csm += self._noise_wishart
        return csm
