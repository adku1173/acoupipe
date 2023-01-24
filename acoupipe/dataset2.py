from os import path
from copy import deepcopy
import numpy as np
from scipy.stats import rayleigh, poisson, norm
from acoular import ImportGrid, MicGeom, \
    Environment, RectGrid3D
from .sampler import CovSampler, NumericAttributeSampler, LocationSampler, MicGeomSampler
from .pipeline import BasePipeline, DistributedPipeline
from .helper import complex_to_real
from .dataset1 import Dataset1
from acoupipe.spectra_analytic import PowerSpectraAnalytic

VERSION = "ds2-v01"
dirpath = path.dirname(path.abspath(__file__))
ap = 1.4648587220804408 # default aperture size

@complex_to_real
def calc_p2(Q,fidx):
    """function to obtain the auto- and cross-power (Pa^2) of each source.

    Parameters
    ----------
    Q : numpy.array
        the source strength covariance matrix
    fidx : None, list
        frequency indices to be included

    Returns
    -------
    numpy.array
        covariance matrix containing the squared sound pressure of each source
    """
    if fidx:
        return np.array([Q[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return Q.copy()

class Dataset2(Dataset1):

    def __init__(
            self, 
            split, 
            size, 
            features, 
            f=None, 
            num=0, 
            fs=51200,
            startsample=1, 
            max_nsources = 10,
            min_nsources = 1,
            env = Environment(c=343.),
            mics = MicGeom(from_file=path.join(dirpath, "xml", "tub_vogel64.xml")),
            grid = RectGrid3D(y_min=-.5*ap,y_max=.5*ap,x_min=-.5*ap,x_max=.5*ap,z_min=.5*ap,z_max=.5*ap,increment=1/63*ap),
            cache_bf = False,
            cache_dir = "./datasets",         
            progress_bar= False,   
            config=None):  
        super().__init__(
                split=split, 
                size=size, 
                features=features, 
                f=f, 
                num=num, 
                fs=fs, 
                startsample=startsample, 
                max_nsources=max_nsources, 
                min_nsources=min_nsources,
                env = env,
                mics = mics,
                grid = grid,
                cache_csm=False,
                cache_bf=cache_bf,
                cache_dir=cache_dir, 
                progress_bar=progress_bar,
                config=config)
        # overwrite freq_data
        fftfreq = abs(np.fft.fftfreq(512, 1./self.fs)[:int(512/2+1)])[1:]          
        self.freq_data = PowerSpectraAnalytic(frequencies=fftfreq)   
        self.random_var = {
        "mic_rvar" : norm(loc=0, scale=0.001), # positional noise on the microphones  
        "p2_rvar" : rayleigh(5),
        "loc_rvar" : (
                    norm((self.grid.x_min + self.grid.x_max)/2,0.1688*(np.sqrt((self.grid.x_max-self.grid.x_min)**2))), #x
                    norm((self.grid.y_min + self.grid.y_max)/2,0.1688*(np.sqrt((self.grid.y_max-self.grid.y_min)**2))), #y
                    norm((self.grid.z_min + self.grid.z_max)/2,0*(np.sqrt((self.grid.z_max-self.grid.z_min)**2)))), #z
        "nsrc_rvar" : poisson(mu=3, loc=1)  # number of sources
        }

    def _prepare(self):
        self.freq_data.steer.grid = ImportGrid(gpos_file=self.loc_sampler.target)
        self.freq_data.Q = self.strength_sampler.target

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
            "p2" : (calc_p2, self.freq_data.Q, fidx),
            }                
        # add input features (csm, sourcemap, cleansc, ...)
        features.update(self.setup_features())
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler,features=features, prepare=self._prepare, progress_bar=self.progress_bar)

    def setup_sampler(self):
        sampler = []
        mic_sampling = MicGeomSampler(
            random_var= self.random_var["mic_rvar"],
            target=self.noisy_mics,
            ddir=np.array([[1.0], [1.0], [1.0]]))  
        sampler.append(mic_sampling)

        self.strength_sampler = CovSampler(
            random_var = self.random_var["p2_rvar"],
            nsources = self.max_nsources,
            nfft = self.freq_data.fftfreq().shape[0])
        sampler.append(self.strength_sampler)

        self.loc_sampler = LocationSampler(
            random_var=self.random_var["loc_rvar"],
            nsources = self.max_nsources,
            x_bounds=(self.grid.x_min, self.grid.x_max),
            y_bounds=(self.grid.y_min, self.grid.y_max),
            z_bounds=(self.grid.z_min, self.grid.z_max))
        sampler.append(self.loc_sampler)
      
        if not (self.max_nsources == self.min_nsources):  # if no number of sources is specified, the number of sources will be samples randomly
            nsrc_sampling = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[self.strength_sampler,self.loc_sampler],
                attribute='nsources',
                single_value = True,
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources))
            sampler = [nsrc_sampling] + sampler
        return sampler