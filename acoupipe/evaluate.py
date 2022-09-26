# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Adam Kujawski, Simon Jekosch, Art Pelling, Ennes Sarradj.
#------------------------------------------------------------------------------

"""
Provides evaluatation of source mapping methods using classes derived from :class:`~acoupipe.evaluate.BaseEvaluator`.


.. autosummary::
    :toctree: generated/

    BaseEvaluator
    PlanarSourceMapEvaluator
    GridlessEvaluator

"""


from acoular.grids import RectGrid
from traits.api import Instance, HasPrivateTraits, CArray, Float, Property, Int, Bool
from acoular import CircSector, RectGrid, L_p, integrate
from warnings import warn
from numpy import searchsorted
import numpy as np

def get_frequency_index_range(freq,f,num):
    """Returns the left and right indices that define the frequency range 
    to integrate over.

    Parameters
    ----------
    freq : numpy.array
        frequency vector (can be determined by evaluating `freqdata()` method at a `acoular.PowerSpectra` instance)
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)

    Returns
    -------
    tuple
        left and right index that belongs to the frequency of interest
    """
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
    """Base class to evaluate the performance of source mapping methods."""
    
    #: the target locations with shape=(ns,ndim). ns is the number of ground-truth sources,
    #: ndim specifies the spatial dimension
    target_loc = CArray(
        desc="locations of the ground-truth sources")  
    
    #: the target power values with shape=(nf,ns). ns is the number of ground-truth sources,
    #: nf specifies the number of frequencies to evaluate
    target_pow = CArray(  
        desc="power of the ground-truth sources")  

    #: size of the integration sector. The true size of the integration area can vary if
    #: attr:`variable_sector_radii` is True and the distance between two sources is smaller
    #: than 2*r.
    r = Float(0.05,
        desc="radius of integration around true source position") 

    #: returns the determined sector sizes for each ground-truth source position
    sector_radii = Property()

    #: if set True: use adaptive integration area
    variable_sector_radii = Bool(True,
        desc="adaptive integration area")

    def _get_sector_radii(self):
        ns = self.target_pow.shape[1]
        if ns > 1: # only if multiple sources are present
            if self.variable_sector_radii:
                radii = [] 
                for i in range(ns):
                    intersourcedists = np.linalg.norm(self.target_loc - self.target_loc[i,:],axis=1)
                    intersourcedists = intersourcedists[intersourcedists != 0] 
                    if intersourcedists.min()/2 < self.r:
                        radii.append(intersourcedists.min()/2)
                    else: 
                        radii.append(self.r)
            else:
                radii = [self.r]*ns
        else:
            radii = [self.r] 
        return radii
    
    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017)"""
        pass

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017)"""
        pass

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017)"""
        pass


class PlanarSourceMapEvaluator(BaseEvaluator):
    """
    Class to evaluate the performance of microphone array methods on planar grid-based source maps.

    This class can be used to calculate different performance metrics
    to assess the performance of a source mapping method, including:
    * specific level error (Herold and Sarradj, 2017)
    * overall level error (Herold and Sarradj, 2017)
    * inverse level error (Herold and Sarradj, 2017)
    """

    #: an array of shape=(nf,nx,ny) containing sourcemaps to evaluate. nf is the number of frequencies,
    #: nx is the number of spatial samples in x-direction, ny is the number of spatial samples in y-direction.
    sourcemap = CArray(
        desc="an array of shape=(nfrequencies,nxsteps,nysteps) containing sourcemaps to evaluate") 

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    grid = Instance(RectGrid,
        desc="beamforming grid instance that belongs to the sourcemap")

    def _integrate_targets(self,multi_assignment=True):
        """integrates over target sectors.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated p^2 values for each region
        """
        results = np.empty(shape=self.target_pow.shape)
        for f in range(self.target_pow.shape[0]):
            pm = self.sourcemap[f].copy()
            for i in range(self.target_pow.shape[1]):
                sector = CircSector(r=self.sector_radii[i],
                                    x=self.target_loc[i, 0],
                                    y=self.target_loc[i, 1])
                results[f,i] = integrate(pm,self.grid,sector)
                if not multi_assignment:
                    indices = self.grid.indices(sector.x,sector.y,sector.r)
                    pm[indices] = 0 # set values to zero (can not be assigned again)
        return results

    def _validate_shapes(self):
        if not self.sourcemap.ndim == 3:
            raise ValueError("attribute sourcemap is not of shape (number of frequencies, nxsteps, nysteps)!")
        if not self.sourcemap.shape[0] == self.target_pow.shape[0]:
            raise ValueError(f"Number of p2 target values per source (shape {self.target_pow.shape}) does not match the number of sourcemaps (shape {self.sourcemap.shape}). Provide as many target values as sourcemaps!")

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,1)
        """
        self._validate_shapes()
        return L_p(self.sourcemap.sum(axis=(1,2))) - L_p(self.target_pow.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources
        """
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result) - L_p(self.target_pow)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)
        """
        self._validate_shapes()
        integration_result = self._integrate_targets(multi_assignment=False) # do not allow to assign the same grid point to multiple sources (otherwise it may result in a positive inverse level error)
        return L_p(integration_result.sum(axis=1)) - L_p(self.sourcemap.sum(axis=(1,2)))


    

class GridlessEvaluator(BaseEvaluator):
    """
    Class to evaluate the performance of microphone array methods on planar grid-less source maps.

    This class can be used to calculate different performance metrics
    to assess the performance of a source mapping method, including:
    * specific level error (Herold and Sarradj, 2017)
    * overall level error (Herold and Sarradj, 2017)
    * inverse level error (Herold and Sarradj, 2017)
    * one-to-one localization error
    * one-to-one level error
    """

    #: the estimated locations with shape=(ns,ndim). ns is the number of sources,
    #: ndim specifies the spatial dimension
    estimated_loc = CArray(
        desc="locations of the estimated sources")  

    #: the estimated power values with shape=(nf,ns). ns is the number of estimated sources,
    #: nf specifies the number of frequencies to evaluate
    estimated_pow = CArray(
        desc="power of the estimated sources")   

    def _validate_shapes(self):
        if not self.estimated_loc.ndim == 2:
            raise ValueError("attribute estimated_loc is not of shape (number of sources, spatial dimension)!")
        if not self.target_loc.ndim == 2:
            raise ValueError("attribute target_loc is not of shape (number of sources, spatial dimension)!")

    def _integrate_targets(self):
        """integrates over target sectors.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated p^2 values for each region
        """
        results = np.empty(shape=self.target_pow.shape)
        for i in range(self.target_pow.shape[1]):
            for f in range(self.target_pow.shape[0]):
                tloc = self.target_loc[i, :]
                eloc = self.estimated_loc[:, :]
                dists = np.sqrt(((tloc-eloc)**2).sum(1))
                integration_mask = dists < self.sector_radii[i]
                results[f,i] = self.estimated_pow[f,integration_mask].sum()
        return results

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,1)
        """
        self._validate_shapes()
        return L_p(self.estimated_pow.sum(axis=1)) - L_p(self.target_pow.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources
        """

        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result) - L_p(self.target_pow)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017)

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)
        """
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result.sum(axis=1)) - L_p(self.estimated_pow.sum(axis=1))

    def get_localization_error(self):
        """Returns the spatial distance between the estimated position and the ground-truth 
        position at the same index 

        Returns
        -------
        numpy.array
            localization error of shape=(ns,). ns: number of estimated sources
        """
        self._validate_shapes()
        return np.linalg.norm(self.estimated_loc-self.target_loc,axis=1)

    def get_level_error(self):
        """Returns the level difference in dB between the estimated power and the ground-truth 
        power at the same index 

        Returns
        -------
        numpy.array
            level error of shape=(ns,). ns: number of estimated sources
        """
        return 10*np.log10(self.estimated_pow) - 10*np.log10(self.target_pow)
