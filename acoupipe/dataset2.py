from copy import deepcopy
from os import path

import numpy as np
from acoular import BeamformerBase, Environment, ImportGrid, MicGeom, RectGrid3D
from scipy.stats import norm, poisson, rayleigh, uniform

from acoupipe.spectra_analytic import PowerSpectraAnalytic

from .dataset1 import Dataset1
from .helper import complex_to_real
from .pipeline import BasePipeline, DistributedPipeline
from .sampler import CovSampler, LocationSampler, MicGeomSampler, NumericAttributeSampler, SpectraSampler

VERSION = "ds2-v01"
ap = 1.4648587220804408 # default aperture size
DEFAULT_ENV = Environment(c=343.)
DEFAULT_MICS = MicGeom(from_file=path.join(path.dirname(path.abspath(__file__)), "xml", "tub_vogel64.xml"))
DEFAULT_GRID = RectGrid3D(y_min=-.5*ap,y_max=.5*ap,x_min=-.5*ap,x_max=.5*ap,z_min=.5*ap,z_max=.5*ap,increment=1/63*ap)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")                   

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
            split, 
            size, 
            features, 
            f=None, 
            num=0, 
            fs=51200,
            max_nsources = 10,
            min_nsources = 1,
            df = 100,
            env = DEFAULT_ENV,
            mics = DEFAULT_MICS,
            grid = DEFAULT_GRID,
            beamformer = DEFAULT_BEAMFORMER,
            sample_noise=False, 
            sample_spectra=False,
            sample_wishart=True):  
        super().__init__(
                split=split, 
                size=size, 
                features=features, 
                f=f, 
                num=num, 
                fs=fs, 
                max_nsources=max_nsources, 
                min_nsources=min_nsources,
                env = env,
                mics = mics,
                grid = grid,
                beamformer = beamformer)
        # overwrite freq_data
        self.sample_spectra = sample_spectra
        self.sample_noise = sample_noise
        fftfreq = abs(np.fft.fftfreq(512, 1./self.fs)[:int(512/2+1)])[1:]          
        self.freq_data = PowerSpectraAnalytic(frequencies=fftfreq, df=df)   
        if sample_wishart:
            self.freq_data.mode = "wishart"
        self.random_var = {
        "mic_rvar" : norm(loc=0, scale=0.001), # positional noise on the microphones  
        "p2_rvar" : rayleigh(5), 
        "loc_rvar" : (
                    norm((self.grid.x_min + self.grid.x_max)/2,0.1688*(np.sqrt((self.grid.x_max-self.grid.x_min)**2))), #x
                    norm((self.grid.y_min + self.grid.y_max)/2,0.1688*(np.sqrt((self.grid.y_max-self.grid.y_min)**2))), #y
                    norm((self.grid.z_min + self.grid.z_max)/2,0*(np.sqrt((self.grid.z_max-self.grid.z_min)**2)))), #z
        "nsrc_rvar" : poisson(mu=3, loc=1),  # number of sources
        "noise_rvar" : uniform(1e-06, 1e-03-1e-06)
        }

    def _prepare(self):
        self.freq_data.steer.grid = ImportGrid(gpos_file=self.loc_sampler.target)
        self.freq_data.Q = self.strength_sampler.target.copy()
        if self.sample_noise:
            self.freq_data.noise = self.noise_sampler.target.copy()

    def build_pipeline(self, parallel=False):
        # create copy for noisy positions
        self.noisy_mics = deepcopy(self.mics) # Microphone geometry with positional noise
        steer_src = deepcopy(self.steer)
        steer_src.mics = self.noisy_mics
        steer_src.ref = self.noisy_mics.mpos[:, self.ref_mic]
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
                nfft = self.freq_data.fftfreq().shape[0])
            sampler.append(self.strength_sampler)

            if self.sample_noise:
                self.noise_sampler = CovSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.mics.num_mics,
                    single_value = True,
                    nfft = self.freq_data.fftfreq().shape[0])
                sampler.append(self.noise_sampler)
        else:
            self.strength_sampler = SpectraSampler(
                random_var = self.random_var["p2_rvar"],
                nsources = self.max_nsources,
                scale_variance = True,
                single_value = False,
                single_spectra = False,
                nfft = self.freq_data.fftfreq().shape[0])
            sampler.append(self.strength_sampler)

            if self.sample_noise:
                self.noise_sampler = SpectraSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.mics.num_mics,
                    single_value = True,
                    single_spectra = True,
                    nfft = self.freq_data.fftfreq().shape[0])
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
