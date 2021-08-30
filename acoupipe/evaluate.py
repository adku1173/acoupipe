from acoular.grids import RectGrid
from traits.api import Instance, HasPrivateTraits, CArray, Float
from acoular import CircSector, RectGrid, L_p, integrate
from warnings import warn
from numpy import searchsorted
import numpy as np

def get_frequency_index_range(freq,f,num):
    """get the summation index"""
    if num == 0:
        # single frequency line
        ind = searchsorted(freq, f)
        if ind >= len(freq):
            warn('Queried frequency (%g Hz) not in resolved '
                            'frequency range. Returning zeros.' % f, 
                            Warning, stacklevel = 2)
            ind = None
        else:
            if freq[ind] != f:
                warn('Queried frequency (%g Hz) not in set of '
                        'discrete FFT sample frequencies. '
                        'Using frequency %g Hz instead.' % (f,freq[ind]), 
                        Warning, stacklevel = 2)
        return (ind,ind+1)
    else:
        # fractional octave band
        if isinstance(num,list):
            f1=num[0]
            f2=num[-1]
        else:
            f1 = f*2.**(-0.5/num)
            f2 = f*2.**(+0.5/num)
        ind1 = searchsorted(freq, f1)
        ind2 = searchsorted(freq, f2)
        if ind1 == ind2:
            warn('Queried frequency band (%g to %g Hz) does not '
                    'include any discrete FFT sample frequencies. '
                    'Returning zeros.' % (f1,f2), 
                    Warning, stacklevel = 2)
        return (ind1,ind2) 


class BaseEvaluator(HasPrivateTraits):

    target_loc = CArray()  # (num_sources, 3)

    target_p2 = CArray()  # (num_sources,num_freq)    

    r = Float(0.05,
        desc="radius of integration around true source position") 

    def get_specific_level_error(self):
        pass

    def get_overall_level_error(self):
        pass

    def get_inverse_level_error(self):
        pass


class PlanarSourceMapEvaluator(BaseEvaluator):

    sourcemap = CArray() # shape=(numfreq,nxsteps,nysteps)

    grid = Instance(RectGrid)

    def _get_sector_radii(self):
        if self.target_p2.shape[1] > 1: # only if multiple sources are present
            radii = [] 
            for i in range(self.target_p2.shape[1]):
                intersourcedists = np.linalg.norm(self.target_loc - self.target_loc[i,:],axis=1)
                intersourcedists = intersourcedists[intersourcedists != 0] 
                if intersourcedists.min()/2 < self.r:
                    radii.append(intersourcedists.min()/2)
                else: 
                    radii.append(self.r)
        else:
            radii = [self.r] 
        return radii

    def _integrate_targets(self):
        """integrates over target sectors.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated p^2 values for each region
        """
        results = np.empty(shape=self.target_p2.shape)
        radii = self._get_sector_radii()
        for i in range(self.target_p2.shape[1]):
            sector = CircSector(r=radii[i],
                                x=self.target_loc[i, 0],
                                y=self.target_loc[i, 1])
            for f in range(self.target_p2.shape[0]):
                results[f,i] = integrate(self.sourcemap[f],self.grid,sector)
        return results

    def _validate_shapes(self):
        if not self.sourcemap.ndim == 3:
            raise ValueError("attribute sourcemap is not of shape (number of frequencies, nxsteps, nysteps)!")
        if not self.sourcemap.shape[0] == self.target_p2.shape[0]:
            raise ValueError(f"Number of p2 target values per source (shape {self.target_p2.shape}) does not match the number of sourcemaps (shape {self.sourcemap.shape}). Provide as many target values as sourcemaps!")

    def get_overall_level_error(self):
        self._validate_shapes()
        return L_p(self.sourcemap.sum(axis=(1,2))) - L_p(self.target_p2.sum(axis=1))

    def get_specific_level_error(self):
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result) - L_p(self.target_p2)

    def get_inverse_level_error(self):
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result.sum(axis=1)) - L_p(self.sourcemap.sum(axis=(1,2)))

