
from acoular import PowerSpectraImport, SteeringVector
from numpy import empty
from traits.api import CArray, Instance, Property, property_depends_on


class PowerSpectraAnalytic(PowerSpectraImport):

    Q = CArray(shape=(None,None,None), dtype=complex, desc="source strengths matrix")

    noise = CArray(shape=(None,None,None), dtype=complex, desc="noise covariance matrix")
    
    steer = Instance(SteeringVector)

    #mode = # wishart, analytic

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")

    def _validate_freq_data ( self ):
        if self.frequencies is None:
            raise ValueError(
                "The frequencies must be given but are None type!")
        else:
            if self.fftfreq().shape[0] != self.Q.shape[0]:
                raise ValueError(
                    "The number of frequencies must match the number of rows in the source strengths matrix!")

    @property_depends_on("Q,steer.digest,ind_low,ind_high")
    def _get_csm ( self ):
        self._validate_freq_data()
        fftfreq = self.fftfreq()
        H = empty((fftfreq.shape[0],self.steer.mics.num_mics,self.Q.shape[1]),dtype=complex)
        for i in self.indices: # calculate only the indices that are needed
            H[i] = self.steer.transfer(fftfreq[i]).T # transfer functions
        csm = H@self.Q@H.swapaxes(2,1).conjugate()
        if self.noise is not None:
            csm += self.noise
        return csm

