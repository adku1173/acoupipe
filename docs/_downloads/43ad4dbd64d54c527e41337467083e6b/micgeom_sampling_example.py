#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy
from numpy.random import RandomState
import matplotlib.pyplot as plt
from acoular import MicGeom
from acoupipe import MicGeomSampler


nsamples = 10

# create Microphone Geometry object
mics = MicGeom(from_file="array64_d0o686.xml")

# define random state
rng = RandomState(seed=1) 

# create instantiate random distribution object
# standard deviation, approx 1/3 min dist between mics (mindist/0.686 = 0.0409885123851)
normal_distribution = scipy.stats.norm(loc=0, scale= 0.004) #scale=0.04/3.)

# create MicGeomSampler object
mgs = MicGeomSampler(random_var=normal_distribution,
                     random_state=rng, 
                     target=mics)

#%%
# =============================================================================
# # first deviate individual microphone positions along x-axis
# =============================================================================
mgs.ddir = np.array([[1.],[0],[0]])

plt.figure()
plt.title("Individual Deviation on x-Axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker='x',s=10,label="true positions")
plt.legend()   
plt.show()

#%%
# =============================================================================
# # second: deviate individual microphone positions along x- and y-axis
# =============================================================================
mgs.ddir = np.array([[1.],[0.5],[0]])

plt.figure()
plt.title("Individual Deviation on x- and y- Axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker='x',s=10,label="true positions")
plt.legend()    
plt.show()

#%%
# =============================================================================
# third: rotate around axis
# =============================================================================

mgs.ddir = np.array([[0.0],[0.0],[0.0]]) # no individual deviation

# for additional rotation around z-axis
mgs.rvec = np.array([[0], [0], [1]])

plt.figure()
plt.title("Rotation Around z-Axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker='x',s=10,label="true positions")
plt.legend()    
plt.show()

#%%
# =============================================================================
# fourth: translate full array along y-axis
# =============================================================================

mgs.rvec = np.array([[0], [0], [0]])
mgs.tdir = np.array([[0], [2.], [0]])

plt.figure()
plt.title("Translation of Full Geometry Along y-Axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker='x',s=10,label="true positions")
plt.legend()    
plt.savefig("output.png",dpi=1200) 
plt.show()

# %%