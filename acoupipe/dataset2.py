from copy import deepcopy
from functools import partial
import numpy as np
from scipy.stats import rayleigh, poisson, norm
from acoular import ImportGrid, SteeringVector, PowerSpectraImport, MicGeom, RectGrid, \
    Environment, RectGrid3D, BeamformerBase, BeamformerCleansc
from .sampler import QSampler, NumericAttributeSampler, LocationSampler, MicGeomSampler
from .pipeline import BasePipeline, DistributedPipeline
from .features import get_sourcemap
from .helper import get_frequency_index_range
from .dataset1 import Dataset as Dataset1

def calc_csm(fidx,fftfreq,steer,Q,loc):
    steer.grid = ImportGrid(gpos_file=loc.target.copy())    
    if fidx:
        csm = np.empty((len(fidx),steer.mics.num_mics,steer.mics.num_mics))
        for k, indices in enumerate(fidx):
            nfreqs = indices[1] - indices[0]
            Q = Q.target[indices[0]:indices[1]]
            H = np.empty((nfreqs,steer.mics.num_mics,loc.target.shape[1]),dtype=complex)
            for i,f in enumerate(fftfreq):
                H[i] = steer.transfer(f).T # transfer functions
            H_h = H.swapaxes(2,1).conjugate() # Hermetian
            csm[k] = (H@Q@H_h).sum(-1)
#            freqs = fftfreq[fidx[0]:fidx[1]]
        return csm
    else: # return the full csm
        Q = Q.target.copy()
#        freqs = fftfreq
        H = np.empty((Q.shape[0],steer.mics.num_mics,loc.target.shape[1]),dtype=complex)
        for i,f in enumerate(fftfreq):
            H[i] = steer.transfer(f).T # transfer functions
        H_h = H.swapaxes(2,1).conjugate() # Hermetian
        return H@Q@H_h

def calc_sourcemap(f,num,b,ps,steer,Q,loc):
    ps.csm = calc_csm(f,steer,Q,loc)
    ps.frequencies = f
    #return b.synthetic(f,num)
    return get_sourcemap(b,[f],num,cache_dir=None,num_threads=1)

def calc_Q(Q,fidx):
    if fidx:
        return Q.target[fidx[0]:fidx[1]]
    else:
        return Q.target.copy()


class Dataset(Dataset1):

    def build_pipeline(self, parallel=False):

        # get config
        c = self.config
        fftfreq = self._fftfreq()[1:] # no zero bin
        if self.f != None:
            if type(self.f) == float or type(self.f) == int:
                self.f = [self.f]
            fidx = [get_frequency_index_range(fftfreq, f_, self.num) for f_ in self.f]
        else:
            fidx = None

        if c["max_nsources"] == c["min_nsources"]:
            nsources_constant = True
        else:
            nsources_constant = False

        sv_args = {'steer_type': c['steer_type'],'env': self.env }                    
        st = SteeringVector(
            grid=self.grid, mics=self.mics, ref=self.mics.mpos[:, c['ref_mic']], **sv_args)        
        st_src = SteeringVector(
            mics=self.noisy_mics, ref=self.noisy_mics.mpos[:, c['ref_mic']], **sv_args)        
        # Set up Beamformer object to calculate sourcemap feature
        if ("sourcemap" in self.features) or ("ref_cleansc" in self.features):
            ps = PowerSpectraImport(csm=np.zeros((1,64,64)),frequencies=0)
            bb_args = {'r_diag': c['r_diag'], 'cached': c['cache_bf'], 'precision': 'float32'}
            bb = BeamformerBase(freq_data=ps, steer=st, **bb_args)
            bfcleansc = BeamformerCleansc(freq_data=ps, steer=st, **bb_args)
                
        ####### sampler #################################
        sampler_list = []

        mic_sampling = MicGeomSampler(
            random_var= norm(loc=0, scale=0.001),
            target=self.noisy_mics,
            ddir=np.array([[1.0], [1.0], [1.0]])
        )  # ddir along two dimensions -> bivariate sampling
        sampler_list.append(mic_sampling)

        strength_sampling = QSampler(
            random_var = rayleigh(5),
            nsources = c["max_nsources"],
            nfft = fftfreq.shape[0]
            )
        sampler_list.append(strength_sampling)

        pos_sampling = LocationSampler(
            random_var=(
                norm((c['x_min'] + c['x_max'])/2,0.1688*(np.sqrt((c['x_max']-c['x_min'])**2))),
                norm((c['y_min'] + c['y_max'])/2,0.1688*(np.sqrt((c['y_max']-c['y_min'])**2))),
                norm((c['z_min'] + c['z_max'])/2,0*(np.sqrt((c['z_max']-c['z_min'])**2)))),
            x_bounds=(c['x_min'], c['x_max']),
            y_bounds=(c['y_min'], c['y_max']),
            z_bounds=(c['z_min'], c['z_max']),
        )
        sampler_list.append(pos_sampling)
      
        if not nsources_constant:  # if no number of sources is specified, the number of sources will be samples randomly
            nsrc_sampling = NumericAttributeSampler(
                random_var=poisson(mu=3, loc=1),
                target=[strength_sampling,pos_sampling],
                attribute='nsources',
                single_value = True,
                filter=lambda x: (x <= c['max_nsources']) and (
                    x >= c['min_nsources']),
            )
            sampler_list = [nsrc_sampling] + sampler_list

        # set up feature methods
        features={
            "loc" : lambda: pos_sampling.target.copy(),
            "Q" : (calc_Q, strength_sampling, fidx),
            }                

        if "csm" in self.features:
            features["csm"] = (calc_csm,fidx,fftfreq,st_src,strength_sampling,pos_sampling)
        if "sourcemap" in self.features:
            features["sourcemap"] = (calc_sourcemap,self.f,self.num,bb,ps,st_src,strength_sampling,pos_sampling)
        elif "ref_cleansc" in self.features:
            features["ref_cleansc"] = (calc_sourcemap,self.f,self.num,bfcleansc,ps,st_src,strength_sampling,pos_sampling)


        # set up pipeline
        if parallel:
            Pipeline = DistributedPipeline
        else:
            Pipeline = BasePipeline
        return Pipeline(sampler=sampler_list,features=features)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    d = Dataset(size=1,f=1000,num=3,features=["csm"],split="test")
    gen = d.generate()
    for d in gen:
        pass




    # d = Dataset1(size=1,f=1000,features=["sourcemap"],split="training")
    # gen = d.generate()
    # for d in gen:
    #     print(d["sourcemap"].shape)
