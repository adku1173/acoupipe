from copy import deepcopy
from os import path

import numpy as np
from acoular import BeamformerBase, Environment, ImportGrid, MicGeom, RectGrid3D, SteeringVector
from scipy.stats import norm, poisson, rayleigh, uniform

from acoupipe.spectra_analytic import PowerSpectraAnalytic

from .dataset1 import Dataset1
from .helper import complex_to_real
from .pipeline import BasePipeline, DistributedPipeline
from .sampler import CovSampler, LocationSampler, MicGeomSampler, NumericAttributeSampler, SpectraSampler

VERSION = "ds2-v01"
DEFAULT_ENV = Environment(c=343.)
DEFAULT_MICS = MicGeom(from_file=path.join(path.dirname(path.abspath(__file__)), "xml", "tub_vogel64.xml"))
ap = DEFAULT_MICS.aperture
DEFAULT_GRID = RectGrid3D(y_min=-.5*ap,y_max=.5*ap,x_min=-.5*ap,x_max=.5*ap,z_min=.5*ap,z_max=.5*ap,increment=1/63*ap)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")                   
DEFAULT_STEER = SteeringVector(grid=DEFAULT_GRID, mics=DEFAULT_MICS, env=DEFAULT_ENV, steer_type ="true level")
DEFAULT_FREQ_DATA = PowerSpectraAnalytic(df=500)
DEFAULT_RANDOM_VAR = {
    "mic_rvar" : norm(loc=0, scale=0.001), # positional noise on the microphones  
    "p2_rvar" : rayleigh(5), 
    "loc_rvar" : (
        norm((DEFAULT_GRID.x_min + DEFAULT_GRID.x_max)/2,0.1688*(np.sqrt((DEFAULT_GRID.x_max-DEFAULT_GRID.x_min)**2))),
        norm((DEFAULT_GRID.y_min + DEFAULT_GRID.y_max)/2,0.1688*(np.sqrt((DEFAULT_GRID.y_max-DEFAULT_GRID.y_min)**2))),
        norm((DEFAULT_GRID.z_min + DEFAULT_GRID.z_max)/2,0*(np.sqrt((DEFAULT_GRID.z_max-DEFAULT_GRID.z_min)**2)))), 
    "nsrc_rvar" : poisson(mu=3, loc=1),  # number of sources
    "noise_rvar" : uniform(1e-06, 1e-03-1e-06) # variance of the noise
    }

@complex_to_real
def calc_p2(freq_data,fidx):
    """Calculates the auto- and cross-power (Pa^2) of each source.

    Parameters
    ----------
    freq_data : AnalyticPowerSpectra
        the source strength covariance matrix
    fidx : None, list
        frequency indices to be included

    Returns
    -------
    numpy.array
        covariance matrix containing the squared sound pressure of each source
    """
    if fidx:
        return np.array([freq_data.Q[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return freq_data.Q.copy()

@complex_to_real
def calc_n2(freq_data,fidx):
    """Calculates the auto- and cross-power (Pa^2) of the noise.

    Parameters
    ----------
    freq_data : AnalyticPowerSpectra
        the source strength covariance matrix
    fidx : None, list
        frequency indices to be included

    Returns
    -------
    numpy.array
        covariance matrix containing the squared sound pressure of each source
    """
    if fidx:
        return np.array([freq_data.noise[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return freq_data.noise.copy()

class Dataset2(Dataset1):

    def __init__(
            self, 
            mics = DEFAULT_MICS,
            grid = DEFAULT_GRID,
            beamformer = DEFAULT_BEAMFORMER,
            steer = DEFAULT_STEER,
            freq_data = DEFAULT_FREQ_DATA,
            random_var = DEFAULT_RANDOM_VAR,
            sample_noise=False, 
            sample_spectra=False,
            sample_wishart=True,
            nfft = 256,
            **kwargs):  
        super().__init__(
                mics = mics,
                grid = grid,
                beamformer = beamformer,
                steer = steer,
                freq_data = freq_data,
                random_var = random_var,
                **kwargs)
        # overwrite freq_data
        self.sample_spectra = sample_spectra
        self.sample_noise = sample_noise
        self.sample_wishart = sample_wishart
        self.nfft = nfft
        if sample_wishart:
            self.freq_data.mode = "wishart"

    def _prepare(self):
        self.freq_data.steer.grid = ImportGrid(gpos_file=self.loc_sampler.target)
        self.freq_data.Q = self.strength_sampler.target.copy()
        if self.sample_noise:
            self.freq_data.noise = self.noise_sampler.target.copy()

    def build_pipeline(self, parallel=False):
        self.freq_data.frequencies=abs(np.fft.fftfreq(self.nfft*2, 1./self.fs)[:int(self.nfft+1)])[1:]   
        # sets the reference microphone to the one closest to the center
        ref_mic = np.argmin(np.linalg.norm((self.mics.mpos - self.mics.center[:,np.newaxis]),axis=0))
        self.steer.ref = self.mics.mpos[:,ref_mic]
        # create copy for noisy positions
        self.noisy_mics = deepcopy(self.mics) # Microphone geometry with positional noise
        steer_src = deepcopy(self.steer)
        steer_src.mics = self.noisy_mics
        steer_src.ref = self.noisy_mics.mpos[:, ref_mic] #TODO: check if this is correct when sampled!
        self.freq_data.steer = steer_src
        # set up sampler
        sampler = self.setup_sampler()
        # set up feature methods
        fidx = self._get_freq_indices()
        features={
            "loc" : lambda: self.loc_sampler.target.copy(),
            "p2" : (calc_p2, self.freq_data, fidx),
            "variances" : lambda: self.strength_sampler.variances.copy(),
            }                
        if self.sample_noise:
            features.update(
                {"nvariances" : lambda: self.noise_sampler.variances.copy(),
                "n2" : (calc_n2, self.freq_data, fidx),})
            
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler,features=features, prepare=self._prepare)

    def setup_sampler(self):
        sampler = [self.freq_data]

        mic_sampler = MicGeomSampler(
            random_var= self.random_var["mic_rvar"],
            target=self.noisy_mics,
            ddir=np.array([[1.0], [1.0], [1.0]]))  
        sampler.append(mic_sampler)

        self.loc_sampler = LocationSampler(
            random_var=self.random_var["loc_rvar"],
            nsources = self.max_nsources,
            x_bounds=(self.grid.x_min, self.grid.x_max),
            y_bounds=(self.grid.y_min, self.grid.y_max),
            z_bounds=(self.grid.z_min, self.grid.z_max))
        sampler.append(self.loc_sampler)

        if not self.sample_spectra:
            self.strength_sampler = CovSampler(
                random_var = self.random_var["p2_rvar"],
                nsources = self.max_nsources,
                scale_variance = True,
                nfft = self.nfft)
            sampler.append(self.strength_sampler)

            if self.sample_noise:
                self.noise_sampler = CovSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.mics.num_mics,
                    single_value = True,
                    nfft = self.nfft)
                sampler.append(self.noise_sampler)
        else:
            self.strength_sampler = SpectraSampler(
                random_var = self.random_var["p2_rvar"],
                nsources = self.max_nsources,
                scale_variance = True,
                single_value = False,
                single_spectra = False,
                nfft = self.nfft)
            sampler.append(self.strength_sampler)

            if self.sample_noise:
                self.noise_sampler = SpectraSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.mics.num_mics,
                    single_value = True,
                    single_spectra = True,
                    nfft = self.nfft)
                sampler.append(self.noise_sampler)
     
        if not (self.max_nsources == self.min_nsources):  
            nsrc_sampler = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[self.strength_sampler,self.loc_sampler],
                attribute="nsources",
                single_value = True,
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources))
            sampler = [nsrc_sampler] + sampler # need to be sampled first!

        return sampler
