"""Provides evaluatation of source mapping methods using classes derived from :class:`~acoupipe.evaluate.BaseEvaluator`."""

import numpy as np
from acoular import CircSector, L_p, integrate 
from acoular.grids import Grid, ImportGrid
from traits.api import Bool, CArray, Float, HasPrivateTraits, Instance, Property
from scipy.spatial.distance import cdist

class BaseEvaluator(HasPrivateTraits):
    """Base class to evaluate the performance of source mapping methods."""
    
    #: the target locations with shape=(ndim, ns). ns is the number of ground-truth sources,
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
        radii = np.ones(ns)*self.r 
        if self.variable_sector_radii:
            intersrcdist = cdist(self.target_loc.T, self.target_loc.T)
            intersrcdist[intersrcdist == 0] = np.inf
            intersrcdist = intersrcdist.min(0)/2
            radii = np.minimum(radii,intersrcdist)
        return radii

    def _get_sectors(self):
        """returns a list of CircSector objects for each target location."""
        if self.target_loc.shape[0] in [2,3]:
            sectors = [
                CircSector(r=self.sector_radii[i],
                                    x=self.target_loc[0,i],
                                    y=self.target_loc[1,i]) 
                for i in range(self.target_pow.shape[1])
            ]
        else:
            raise ValueError(
                "target_loc must have shape (2,nsources)! or (3,nsources) for 3D!")
        return sectors

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017)."""
        pass

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017)."""
        pass

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017)."""
        pass


class SourceMapEvaluator(BaseEvaluator):
    """Class to evaluate the performance of microphone array methods on planar grid-based source maps.

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
    grid = Instance(Grid,
        desc="beamforming grid instance that belongs to the sourcemap")

    def _integrate_targets(self,multi_assignment=True):
        """integrates over target sectors.

        Parameters
        ----------
        multi_assignment : bool, optional
            if set True, the same amplitude can be assigned to multiple targets if 
            the integration area overlaps. The default is True.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated p^2 values for each region
        """
        results = np.empty(shape=self.target_pow.shape)
        sectors = self._get_sectors()
        for f in range(self.target_pow.shape[0]):
            pm = self.sourcemap[f].copy()
            for i in range(self.target_pow.shape[1]):
                results[f,i] = integrate(pm,self.grid,sectors[i])
                if not multi_assignment:
                    indices = self.grid.subdomain(sectors[i])
                    pm[indices] = 0 # set values to zero (can not be assigned again)
        return results

    def _validate_shapes(self):
        if self.grid.shape != self.sourcemap.shape[1:]:
            raise ValueError("grid and sourcemap do not have the same shape!")
        if not self.sourcemap.shape[0] == self.target_pow.shape[0]:
            raise ValueError(
                f"""Number of specified frequencies must match between the sourcemaps and the target power values!
                (shape {self.target_pow.shape[0]}) does not match the number of sourcemaps (shape {self.sourcemap.shape[0]}). 
                Provide as many target values as sourcemaps!""")
        if self.target_loc.shape[0] > 3:
            raise ValueError("target_loc cannot have more than 3 dimensions! Maybe you want to use a transposed target_loc array?")

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,1)
        """
        self._validate_shapes()
        sum_axis = tuple([i for i in range(1,len(self.sourcemap.shape))])
        return L_p(self.sourcemap.sum(axis=sum_axis)) - L_p(self.target_pow.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources
        """
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result) - L_p(self.target_pow)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)
        """
        self._validate_shapes()
        integration_result = self._integrate_targets(multi_assignment=False) 
        sum_axis = tuple([i for i in range(1,len(self.sourcemap.shape))])
        return L_p(integration_result.sum(axis=1)) - L_p(self.sourcemap.sum(axis=sum_axis))


    

class GridlessEvaluator(BaseEvaluator):
    """Class to evaluate the performance of microphone array methods on planar grid-less source maps.

    This class can be used to calculate different performance metrics
    to assess the performance of a source mapping method, including:
    * specific level error (Herold and Sarradj, 2017)
    * overall level error (Herold and Sarradj, 2017)
    * inverse level error (Herold and Sarradj, 2017)
    """

    #: the estimated locations with shape=(ndim, ns). ns is the number of sources,
    #: ndim specifies the spatial dimension
    estimated_loc = CArray(
        desc="locations of the estimated sources")  

    #: the estimated power values with shape=(nf,ns). ns is the number of estimated sources,
    #: nf specifies the number of frequencies to evaluate
    estimated_pow = CArray(
        desc="power of the estimated sources")   

    def _validate_shapes(self):
        if not self.estimated_loc.ndim == 2:
            raise ValueError("attribute estimated_loc is not of shape (spatial dimension, number of sources)!")
        if not self.target_loc.ndim == 2:
            raise ValueError("attribute target_loc is not of shape (spatial dimension, number of sources)!")
        if not self.target_pow.shape[1] == self.estimated_pow.shape[1]:
            raise ValueError("number of target power values does not match the number of estimated source powers!")

    def _integrate_targets(self,multi_assignment=True):
        """integrates over target sectors.

        Parameters
        ----------
        multi_assignment : bool, optional
            if set True, the same amplitude can be assigned to multiple targets if 
            the integration area overlaps. The default is True.

        Returns
        -------
        array (num_freqs,num_sources)
            returns the integrated p^2 values for each region
        """
        results = np.empty(shape=self.target_pow.shape)
        grid = ImportGrid(gpos_file=self.target_loc)
        sectors = self._get_sectors()
        for f in range(self.target_pow.shape[0]):
            pm = self.estimated_pow[f].copy()
            for i in range(self.target_pow.shape[1]):
                results[f,i] = integrate(pm,grid,sectors[i])
                if not multi_assignment:
                    indices = grid.subdomain(sectors[i])
                    pm[indices] = 0 # set values to zero (can not be assigned again)
        return results

    def get_overall_level_error(self):
        """Returns the overall level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            overall level error of shape=(nf,1)
        """
        self._validate_shapes()
        return L_p(self.estimated_pow.sum(axis=1)) - L_p(self.target_pow.sum(axis=1))

    def get_specific_level_error(self):
        """Returns the specific level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            specific level error of shape=(nf,ns). nf: number of frequencies, ns: number of sources
        """
        self._validate_shapes()
        integration_result = self._integrate_targets()
        return L_p(integration_result) - L_p(self.target_pow)

    def get_inverse_level_error(self):
        """Returns the inverse level error (Herold and Sarradj, 2017).

        Returns
        -------
        numpy.array
            inverse level error of shape=(nf,1)
        """
        self._validate_shapes()
        integration_result = self._integrate_targets(multi_assignment=False) 
        return L_p(integration_result.sum(axis=1)) - L_p(self.estimated_pow.sum(axis=1))

