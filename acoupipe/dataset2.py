from os import path
from copy import deepcopy
from functools import partial
import numpy as np
from scipy.stats import rayleigh, poisson, norm
from acoular import ImportGrid, SteeringVector, PowerSpectraImport, MicGeom, RectGrid, \
    Environment, RectGrid3D, BeamformerBase, BeamformerCleansc
from .sampler import CovSampler, NumericAttributeSampler, LocationSampler, MicGeomSampler
from .pipeline import BasePipeline, DistributedPipeline
from .features import get_sourcemap, AnalyticCSMFeature
from .helper import get_frequency_index_range
from .dataset1 import Dataset1

dirpath = path.dirname(path.abspath(__file__))
ap = 1.4648587220804408 # default aperture size

def calc_p2(p2,fidx):
    if fidx:
        return np.array([p2.target[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return p2.target.copy()

config2 = {
    'version': "ds2-v01",  # data set version
    'max_nsources': 10,
    'min_nsources': 1,
    'c': 343.,  # speed of sound
    'fs': 51200,  
    'blocksize': 512,  # block size used for FFT
    'overlap': '50%',
    'window': 'Hanning',
    'r_diag' : False,
    'T': 10,  # length of the simulated signal
    'micgeom': path.join(dirpath, "xml", "tub_vogel64.xml"),  # microphone positions
    'z_min' : 0.5*ap,
    'z_max' : 0.5*ap,
    'y_max' : .5*ap,
    'y_min' : -.5*ap,
    'x_max' : .5*ap,
    'x_min' : -.5*ap,
    'increment' : 1/63*ap,
    'z' : 0.5*ap,
    'ref_mic': 63,  # most center microphone
    'r': 0.05*ap,  # integration radius for sector integration
    'steer_type': 'true level',
    'cache_csm': False,
    'cache_bf': False,
    'cache_dir': "./datasets",
}


class Dataset2(Dataset1):

    def __init__(self, split, size, features, f=None, num=0, startsample=1, config=config2):
        super().__init__(split, size, features, f, num, startsample, config=config)
        self.steer_src = SteeringVector(
            mics=self.noisy_mics, env=self.env, steer_type = config['steer_type'], ref=self.noisy_mics.mpos[:, config['ref_mic']])            
        self.random_var = {
        "mic_rvar" : norm(loc=0, scale=0.001), # positional noise on the microphones  
        "p2_rvar" : rayleigh(5),
        "loc_rvar" : (
                    norm((config['x_min'] + config['x_max'])/2,0.1688*(np.sqrt((config['x_max']-config['x_min'])**2))), #x
                    norm((config['y_min'] + config['y_max'])/2,0.1688*(np.sqrt((config['y_max']-config['y_min'])**2))), #y
                    norm((config['z_min'] + config['z_max'])/2,0*(np.sqrt((config['z_max']-config['z_min'])**2)))), #z
        "nsrc_rvar" : poisson(mu=3, loc=1)  # number of sources
        }

    def _fftfreq ( self ):
        return abs(np.fft.fftfreq(self.config['blocksize'], 1./self.config['fs'])\
                    [:int(self.config['blocksize']/2+1)])[1:]

    def _prepare(self, power_spectra, steer, loc_sampler, strength_sampler):
        steer.grid = ImportGrid(gpos_file=loc_sampler.target)
        H = np.empty((self.strength_sampler.target.shape[0],steer.mics.num_mics,self.strength_sampler.target.shape[1]),dtype=complex)
        for i,f in enumerate(self._fftfreq()):
            H[i] = steer.transfer(f).T # transfer functions
        H_h = H.swapaxes(2,1).conjugate() # Hermitian
        power_spectra.csm = H@strength_sampler.target@H_h

    def build_pipeline(self, parallel=False):

        # setup power spectra Import
        self.ps_csm = PowerSpectraImport(
            csm=np.zeros((self._fftfreq().shape[0],64,64)),
            frequencies=self._fftfreq())

        # set up sampler
        sampler = self.setup_sampler()

        # set up feature methods
        fidx = self._get_freq_indices()
        features={
            "loc" : lambda: self.loc_sampler.target.copy(),
            "p2" : (calc_p2, self.strength_sampler, fidx),
            }                

        # add input features (csm, sourcemap, cleansc, ...)
        features.update(self.setup_features())

        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler,features=features, prepare=partial(self._prepare, self.ps_csm, self.steer_src, self.loc_sampler, self.strength_sampler))

    def setup_sampler(self):
        sampler = []
        mic_sampling = MicGeomSampler(
            random_var= self.random_var["mic_rvar"],
            target=self.noisy_mics,
            ddir=np.array([[1.0], [1.0], [1.0]]))  
        sampler.append(mic_sampling)

        self.strength_sampler = CovSampler(
            random_var = self.random_var["p2_rvar"],
            nsources = self.config["max_nsources"],
            nfft = self._fftfreq().shape[0])
        sampler.append(self.strength_sampler)

        self.loc_sampler = LocationSampler(
            random_var=self.random_var["loc_rvar"],
            x_bounds=(self.config['x_min'], self.config['x_max']),
            y_bounds=(self.config['y_min'], self.config['y_max']),
            z_bounds=(self.config['z_min'], self.config['z_max']))
        sampler.append(self.loc_sampler)
      
        if not (self.config['max_nsources'] == self.config['min_nsources']):  # if no number of sources is specified, the number of sources will be samples randomly
            nsrc_sampling = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[self.strength_sampler,self.loc_sampler],
                attribute='nsources',
                single_value = True,
                filter=lambda x: (x <= self.config['max_nsources']) and (
                    x >= self.config['min_nsources']))
            sampler = [nsrc_sampling] + sampler
        return sampler