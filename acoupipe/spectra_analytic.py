
from acoular import PowerSpectraImport, SteeringVector
from traits.api import  CArray, Instance, Property, property_depends_on
from numpy import empty

class PowerSpectraAnalytic(PowerSpectraImport):

    Q = CArray(shape=(None,None,None), dtype=complex, desc="source strengths matrix")
    
    steer = Instance(SteeringVector)

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

    @property_depends_on('Q,steer.digest')
    def _get_csm ( self ):
        self._validate_freq_data()
        fftfreq = self.fftfreq()
        H = empty((fftfreq.shape[0],self.steer.mics.num_mics,self.Q.shape[1]),dtype=complex)
        for i,f in enumerate(fftfreq):
            H[i] = self.steer.transfer(f).T # transfer functions
        return H@self.Q@H.swapaxes(2,1).conjugate()
