import numpy as np
from scipy.stats import poisson

nsrc_rvar = poisson(mu=3,loc=1) # number of sources

i = 1 # index in sampler list
startindex=0
soff = int(1e7)
#off = int(1e9) # validation
off = 0 # training
numsamples=500000
seed_range = range(off+(i*soff)+startindex, off+(i*soff)+numsamples+startindex)

nsources_list = []
for i,seed in enumerate(seed_range):
    if i%100000 == 0: print(i)
    nsrc_rvar.random_state = np.random.default_rng(seed)     
    nsources_list.append(nsrc_rvar.rvs())

print(max(nsources_list)) # max number of sources within 2 Mio. cases -> 16 sources

