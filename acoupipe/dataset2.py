from copy import deepcopy
from functools import partial

import ray
from acoular import BeamformerBase, Environment, ImportGrid, MicGeom, RectGrid3D, SteeringVector
from numpy import argmin, array, fft, linalg, newaxis, sqrt
from numpy.random import RandomState
from scipy.stats import norm, poisson, rayleigh, uniform

from acoupipe.spectra_analytic import PowerSpectraAnalytic

from .dataset1 import Dataset1, calc_input_features
from .helper import _handle_cache, complex_to_real
from .pipeline import BasePipeline, DistributedPipeline
from .sampler import CovSampler, LocationSampler, MicGeomSampler, NumericAttributeSampler, SpectraSampler

VERSION = "ds2-v01"
DEFAULT_ENV = Environment(c=343.)
mpos = array(
    [[ 0.17166922,  0.27137782,  0.42934454,  0.55315335,  0.70508602,
         0.57496996,  0.42750231,  0.27354702,  0.12063113,  0.06875681,
         0.19167213,  0.3435895 ,  0.50976938,  0.35505328,  0.18113903,
         0.03818231, -0.0436734 , -0.16536651, -0.38907449, -0.59807539,
        -0.41999831, -0.23532124, -0.06110824, -0.09709935, -0.24258568,
        -0.39344719, -0.58015138, -0.68330319, -0.51391496, -0.30056926,
        -0.15456035, -0.00139937, -0.09954774, -0.23581334, -0.40183946,
        -0.57426761, -0.70369547, -0.55371746, -0.43840078, -0.28347743,
        -0.10687031, -0.011395  , -0.12275533, -0.24253211, -0.39588653,
        -0.44323165, -0.27145334, -0.07506058,  0.05903998,  0.17135506,
         0.276813  ,  0.49766801,  0.68590221,  0.50528141,  0.32854486,
         0.14473443,  0.16673884,  0.31969548,  0.45951043,  0.64067241,
         0.35803865,  0.23217426,  0.03926982,  0.05881818],
       [ 0.01705654,  0.13081672,  0.10178682,  0.00905749,  0.10664856,
         0.22071266,  0.28187805,  0.28127827,  0.19453844,  0.33083681,
         0.42769618,  0.46368068,  0.44411011,  0.64382659,  0.61243733,
         0.53562869,  0.69696207,  0.57728417,  0.53459648,  0.40175511,
         0.37229125,  0.4355128 ,  0.41777746,  0.25668547,  0.28120872,
         0.21948553,  0.22843449,  0.01705659,  0.0822511 ,  0.09049545,
         0.11570217,  0.12173658, -0.11120658, -0.06630711, -0.02321287,
        -0.11130545, -0.23725996, -0.33311835, -0.1894243 , -0.20420441,
        -0.26758029, -0.38181978, -0.48840312, -0.35726931, -0.37476845,
        -0.54917305, -0.54427136, -0.66442219, -0.56613574, -0.70739652,
        -0.56715542, -0.47928739, -0.30337791, -0.30991697, -0.41430055,
        -0.43629023, -0.28024325, -0.26060845, -0.15933552, -0.12989189,
        -0.04221796, -0.11721387, -0.20803574, -0.06274283],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ]]
 )

DEFAULT_MICS = MicGeom(mpos_tot=mpos)
#DEFAULT_MICS = MicGeom(from_file=path.join(path.split(acoupipe_path)[0], "xml", "tub_vogel64.xml"))
ap = DEFAULT_MICS.aperture
ref_mic_idx = argmin(linalg.norm((DEFAULT_MICS.mpos - DEFAULT_MICS.center[:,newaxis]),axis=0))
DEFAULT_GRID = RectGrid3D(y_min=-.5*ap,y_max=.5*ap,x_min=-.5*ap,x_max=.5*ap,z_min=.5*ap,z_max=.5*ap,increment=1/63*ap)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")                   
DEFAULT_STEER = SteeringVector(grid=DEFAULT_GRID, mics=DEFAULT_MICS, env=DEFAULT_ENV, steer_type ="true level", 
                               ref=DEFAULT_MICS.mpos[:,ref_mic_idx])
DEFAULT_FREQ_DATA = PowerSpectraAnalytic(df=500)
DEFAULT_RANDOM_VAR = {
    "mic_rvar" : norm(loc=0, scale=0.001), # positional noise on the microphones  
    "p2_rvar" : rayleigh(5), 
    "loc_rvar" : (
        norm((DEFAULT_GRID.x_min + DEFAULT_GRID.x_max)/2,0.1688*(sqrt((DEFAULT_GRID.x_max-DEFAULT_GRID.x_min)**2))),
        norm((DEFAULT_GRID.y_min + DEFAULT_GRID.y_max)/2,0.1688*(sqrt((DEFAULT_GRID.y_max-DEFAULT_GRID.y_min)**2))),
        norm((DEFAULT_GRID.z_min + DEFAULT_GRID.z_max)/2,0*(sqrt((DEFAULT_GRID.z_max-DEFAULT_GRID.z_min)**2)))), 
    "nsrc_rvar" : poisson(mu=3, loc=1),  # number of sources
    "noise_rvar" : uniform(1e-06, 1e-03-1e-06) # variance of the noise
    }


class Dataset2(Dataset1):

    def __init__(
            self, 
            beamformer = DEFAULT_BEAMFORMER,
            steer = DEFAULT_STEER,
            freq_data = DEFAULT_FREQ_DATA,
            random_var = DEFAULT_RANDOM_VAR,
            sample_mic_noise=True,
            sample_noise=False, 
            sample_spectra=False,
            sample_wishart=True,
            nfft = 256,
            **kwargs):  
        super().__init__(
                beamformer = beamformer,
                steer = steer,
                freq_data = freq_data,
                random_var = random_var,
                **kwargs)
        self.sample_mic_noise = sample_mic_noise
        self.sample_spectra = sample_spectra
        self.sample_noise = sample_noise
        self.sample_wishart = sample_wishart
        self.nfft = nfft
        if sample_wishart:
            self.freq_data.mode = "wishart"

    def build_sampler(self):
        sampler = {}
        if self.sample_mic_noise:
            mic_sampler = MicGeomSampler(
                random_var= self.random_var["mic_rvar"],
                target=deepcopy(self.steer.mics),
                ddir=array([[1.0], [1.0], [1.0]]))  
            sampler[1] = mic_sampler

        loc_sampler = LocationSampler(
            random_var=self.random_var["loc_rvar"],
            nsources = self.max_nsources,
            x_bounds=(self.steer.grid.x_min, self.steer.grid.x_max),
            y_bounds=(self.steer.grid.y_min, self.steer.grid.y_max),
            z_bounds=(self.steer.grid.z_min, self.steer.grid.z_max))
        sampler[2] = loc_sampler

        if not self.sample_spectra:
            strength_sampler = CovSampler(
                random_var = self.random_var["p2_rvar"],
                nsources = self.max_nsources,
                scale_variance = True,
                nfft = self.nfft)
            sampler[3] = strength_sampler

            if self.sample_noise:
                noise_sampler = CovSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.steer.mics.num_mics,
                    single_value = True,
                    nfft = self.nfft)
                sampler[4] = noise_sampler
        else:
            strength_sampler = SpectraSampler(
                random_var = self.random_var["p2_rvar"],
                nsources = self.max_nsources,
                scale_variance = True,
                single_value = False,
                single_spectra = False,
                nfft = self.nfft)
            sampler[3] = strength_sampler

            if self.sample_noise:
                noise_sampler = SpectraSampler(
                    random_var = self.random_var["noise_rvar"],
                    nsources = self.steer.mics.num_mics,
                    single_value = True,
                    single_spectra = True,
                    nfft = self.nfft)
                sampler[4] = noise_sampler
     
        if not (self.max_nsources == self.min_nsources):  
            sampler[0] = NumericAttributeSampler(
                random_var=self.random_var["nsrc_rvar"],
                target=[strength_sampler,loc_sampler],
                attribute="nsources",
                single_value = True,
                filter=lambda x: (x <= self.max_nsources) and (
                    x >= self.min_nsources))
        
        if self.sample_wishart:
            sampler[5] = RandomState()
        return sampler

    def build_pipeline(self, parallel, cache_csm, cache_bf, cache_dir):
        cache_dir = _handle_cache(cache_bf, cache_csm, cache_dir)
        ref_mic_idx = argmin(linalg.norm((self.steer.mics.mpos - self.steer.mics.center[:,newaxis]),axis=0))
        self.freq_data.cached = cache_csm
        self.beamformer.cached = cache_bf
        # the frequencies of the spectra
        self.freq_data.frequencies=abs(fft.fftfreq(self.nfft*2, 1./self.fs)[:int(self.nfft+1)])[1:]   
        # the steering vector of the sources (holds the noisy data)
        self.beamformer.steer = deepcopy(self.steer)
        self.freq_data.steer = deepcopy(self.steer) # different from the one in the beamformer
        # set up sampler
        sampler = self.build_sampler()
        # set up feature methods
        fidx = self._get_freq_indices()
        if fidx is not None:
            # bound calculated frequencies for efficiency reasons
            self.freq_data.ind_low = min([f[0] for f in fidx])
            self.freq_data.ind_high = max([f[1] for f in fidx])           
        # set up pipeline
        if parallel:
            freq_data = ray.put(self.freq_data) # already put in the object store
            beamformer = ray.put(self.beamformer)
            Pipeline = DistributedPipeline
        else:
            freq_data = self.freq_data
            beamformer = self.beamformer
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler, 
                        features=(
                            partial(calc_features,
                                fidx = fidx,
                                f = self.f,
                                num = self.num,
                                cache_bf = cache_bf,
                                cache_csm = cache_csm,
                                cache_dir = cache_dir,
                                ref_mic_idx=ref_mic_idx), 
                            self.features, freq_data, beamformer))

def calc_features(s, input_features, freq_data, beamformer, fidx, f, num, cache_bf, cache_csm, cache_dir, ref_mic_idx):
    mic_sampler = s.get(1)
    if mic_sampler is not None:
        freq_data.steer.mics = s[1].target # use noisy microphone positions for measurement
        freq_data.steer.ref = s[1].target.mpos[:, ref_mic_idx] 
    freq_data.steer.grid = ImportGrid(gpos_file=s[2].target)
    freq_data.Q = s[3].target
    noise_sampler = s.get(4)
    if noise_sampler is not None:
        freq_data.noise = noise_sampler.target
    wishart_sampler = s.get(5)
    if wishart_sampler is not None:
        freq_data.seed = wishart_sampler.get_state()[1][0] # the current seed
    beamformer.freq_data = freq_data # change the freq_data, but not the steering!
    # get features
    data = {"loc" : s[2].target,
            "p2" : calc_p2(freq_data, fidx),
            "variances" : s[3].variances}
    if noise_sampler is not None:
        data.update({"nvariances" : noise_sampler.variances,
                    "n2" : calc_n2(freq_data, fidx)})
    data.update(
        calc_input_features(input_features, freq_data, beamformer, fidx, f, num, cache_bf, cache_csm, cache_dir)
    )
    return data

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
        return array([freq_data.Q[indices[0]:indices[1]].sum(0) for indices in fidx])
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
        return array([freq_data.noise[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return freq_data.noise.copy()
