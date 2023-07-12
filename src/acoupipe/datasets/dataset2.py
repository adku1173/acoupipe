from copy import deepcopy
from functools import partial

from acoular import BeamformerBase, Environment, ImportGrid, MicGeom, RectGrid3D, SteeringVector
from numpy import argmin, array, fft, linalg, newaxis, sqrt
from numpy.random import RandomState
from scipy.stats import norm, poisson, rayleigh, uniform

from acoupipe.datasets.dataset1 import Dataset1, calc_input_features
from acoupipe.datasets.helper import _handle_cache, complex_to_real
from acoupipe.datasets.micgeom import tub_vogel64
from acoupipe.datasets.spectra_analytic import PowerSpectraAnalytic
from acoupipe.pipeline import BasePipeline, DistributedPipeline
from acoupipe.sampler import CovSampler, LocationSampler, MicGeomSampler, NumericAttributeSampler, SpectraSampler

VERSION = "ds2-v01"
DEFAULT_ENV = Environment(c=343.)
DEFAULT_MICS = MicGeom(mpos_tot=tub_vogel64)
ap = DEFAULT_MICS.aperture
ref_mic_idx = argmin(linalg.norm((DEFAULT_MICS.mpos - DEFAULT_MICS.center[:,newaxis]),axis=0))
DEFAULT_GRID = RectGrid3D(y_min=-.5*ap,y_max=.5*ap,x_min=-.5*ap,x_max=.5*ap,z_min=.5*ap,z_max=.5*ap,increment=1/63*ap)
DEFAULT_BEAMFORMER = BeamformerBase(r_diag = False, precision = "float32")                   
DEFAULT_STEER = SteeringVector(grid=DEFAULT_GRID, mics=DEFAULT_MICS, env=DEFAULT_ENV, steer_type ="true level", 
                               ref=DEFAULT_MICS.mpos[:,ref_mic_idx])
DEFAULT_FREQ_DATA = PowerSpectraAnalytic(df=500, steer=DEFAULT_STEER)
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
            fs = 51200,
            beamformer = DEFAULT_BEAMFORMER,
            steer = DEFAULT_STEER,
            freq_data = DEFAULT_FREQ_DATA,
            random_var = DEFAULT_RANDOM_VAR,
            sample_mic_noise=True,
            sample_noise=False, 
            sample_spectra=True,
            sample_wishart=True,
            nfft = 256,
            **kwargs):  
        super().__init__(
                fs = fs,
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

    def get_dataset_metadata(self):
        metadata = {}
        metadata["features"] = "-".join(self.features)
        if self.f is not None:
            metadata["f"] = "-".join(map(str,self.f))
        else:
            metadata["f"] = "all"
        metadata["num"] = self.num
        metadata["fs"] = self.fs
        metadata["nfft"] = self.nfft
        metadata["max_nsources"] = self.max_nsources
        metadata["min_nsources"] = self.min_nsources
        metadata["sample_mic_noise"] = self.sample_mic_noise
        metadata["sample_noise"] = self.sample_noise
        metadata["sample_spectra"] = self.sample_spectra
        metadata["sample_wishart"] = self.sample_wishart
        metadata["version"] = VERSION
        return metadata

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
        if self.sample_wishart:
            self.freq_data.mode = "wishart"
        else:
            self.freq_data.mode = "analytic"
        self.freq_data.cached = False # is always false for PowerSpectraAnalytic (Not implemented so far)
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
        else:
            self.freq_data.ind_low = 10 # TODO: not clear why freq_data needs to be set here (otherwise ind_low stays None, seems to be an issue with the digest and property attributes)
            self.freq_data.ind_low = 0
            self.freq_data.ind_high = None
        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler, 
                        features=partial(calc_features,
                                freq_data=self.freq_data,
                                beamformer=self.beamformer,
                                input_features=self.features,
                                fidx=fidx,
                                f=self.f,
                                num=self.num,
                                cache_bf = cache_bf,
                                cache_csm = cache_csm,
                                cache_dir = cache_dir,
                                ref_mic_idx=ref_mic_idx))

    def get_feature_shapes(self):
        sampler = self.build_sampler() # number of samplers
        sdim = len(sampler.values())
        del sampler
        self.freq_data.frequencies=abs(
            fft.fftfreq(self.nfft*2, 1./self.fs)[:int(self.nfft+1)])[1:]
        fidx = self._get_freq_indices() # number of frequencies
        if fidx is None:
            fdim = self.nfft
        else: 
            fdim = len(fidx)
        if self.max_nsources == self.min_nsources: # number of sources
            ndim = self.max_nsources
        else:
            ndim = None
        mdim = self.steer.mics.num_mics # number of microphones
        features_shapes = {
            "idx" : (),
            "seeds" : (sdim,2),
            "loc" : (3,ndim),
            "p2" : (fdim,ndim,ndim,2),
            "variances" : (ndim,)}
        if self.sample_noise:
            features_shapes.update({"n2" : (fdim,mdim,mdim,2), "nvariances" : (mdim,)})
        if self.sample_wishart:
            features_shapes.update({"p2_wishart" : (fdim,ndim,ndim,2)})
            if self.sample_noise:
                features_shapes.update({"n2_wishart" : (fdim,mdim,mdim,2)})
        if "csm" in self.features:
            features_shapes.update({"csm" : (fdim,mdim,mdim,2)})
        if "csmtriu" in self.features:
            features_shapes.update({"csmtriu" : (fdim,mdim,mdim,1)})
        if "sourcemap" in self.features:
            features_shapes.update({"sourcemap" : (fdim,) + self.steer.grid.shape })
        if "eigmode" in self.features:
            features_shapes.update({"eigmode" : (fdim,mdim,mdim,2) })
        return features_shapes


def calc_features(s, freq_data, beamformer, input_features, fidx, f, num, cache_bf, cache_csm, cache_dir, ref_mic_idx):
    # get samplers
    mic_sampler = s.get(1)
    loc_sampler = s.get(2)
    strength_sampler = s.get(3)
    noise_sampler = s.get(4)
    wishart_sampler = s.get(5)

    # update dependent objects with sampled values
    if mic_sampler is not None:
        freq_data.steer.mics = mic_sampler.target # use noisy microphone positions for measurement
        freq_data.steer.ref = mic_sampler.target.mpos[:, ref_mic_idx] 
    freq_data.steer.grid = ImportGrid(gpos_file=loc_sampler.target.copy())
    freq_data.Q = strength_sampler.target.copy()
    if noise_sampler is not None:
        freq_data.noise = noise_sampler.target.copy()
    else:
        freq_data.noise = None
    if wishart_sampler is not None:
        freq_data.seed = wishart_sampler.get_state()[1][0] # the current seed
    beamformer.freq_data = freq_data # change the freq_data, but not the steering!
    
    # get features
    data = {"loc" : loc_sampler.target,
            "p2" : calc_cov(freq_data.Q, fidx),
            "variances" : strength_sampler.variances}
    data.update(
        calc_input_features(input_features, freq_data, beamformer, fidx, f, num, cache_bf, cache_csm, cache_dir)
    )
    if noise_sampler is not None:
        data.update({"nvariances" : noise_sampler.variances,
                    "n2" : calc_cov(freq_data.noise, fidx)})
    if wishart_sampler is not None:
        data.update({"p2_wishart" : calc_cov(freq_data._Q_wishart, fidx)})
        if noise_sampler is not None:
            data.update({"n2_wishart" : calc_cov(freq_data._noise_wishart, fidx)})
    return data

@complex_to_real
def calc_cov(cov, fidx):
    """Return the auto- and cross-power (Pa^2) of each source.

    Parameters
    ----------
    cov : numpy.array
        the strength covariance matrix of shape (nfft, n, n)
    fidx : None, list
        frequency indices to be included

    Returns
    -------
    numpy.array
        covariance matrix containing the squared sound pressure of the respective frequencies
    """
    if fidx:
        return array([cov[indices[0]:indices[1]].sum(0) for indices in fidx])
    else:
        return cov.copy()
