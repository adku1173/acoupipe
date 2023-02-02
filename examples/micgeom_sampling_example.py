import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.random import RandomState

from acoupipe.datasets.dataset1 import DEFAULT_MICS
from acoupipe.sampler import MicGeomSampler

nsamples = 10

# create Microphone Geometry object
mics = DEFAULT_MICS
# define random state
rng = RandomState(seed=1) 

# create instantiate random distribution object
# standard deviation, approx 1/3 min dist between mics (mindist/0.686 = 0.0409885123851)
normal_distribution = scipy.stats.norm(loc=0, scale= 0.004) #scale=0.04/3.)

# create MicGeomSampler object
mgs = MicGeomSampler(random_var=normal_distribution,
                     random_state=rng, 
                     target=mics)


# =============================================================================
# # first deviate individual microphone positions along x-axis
# =============================================================================
mgs.ddir = np.array([[1.],[0],[0]])

plt.figure()
plt.title("individual deviation on x-axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker="x",s=10,label="true positions")
plt.legend()    
plt.show()


# =============================================================================
# # second: deviate individual microphone positions along x- and y-axis
# =============================================================================
mgs.ddir = np.array([[1.],[0.5],[0]])

plt.figure()
plt.title("individual deviation on x- and y- axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker="x",s=10,label="true positions")
plt.legend()    
plt.show()


# =============================================================================
# third: rotate around axis
# =============================================================================

mgs.ddir = np.array([[0.0],[0.0],[0.0]]) # no individual deviation

# for additional rotation around z-axis
mgs.rvec = np.array([[0], [0], [1]])

plt.figure()
plt.title("rotation around z-axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker="x",s=10,label="true positions")
plt.legend()    
plt.show()


# =============================================================================
# fourth: translate full array along y-axis
# =============================================================================

mgs.rvec = np.array([[0], [0], [0]])
mgs.tdir = np.array([[0], [2.], [0]])

plt.figure()
plt.title("translation of full geometry along y-axis")
for _ in range(nsamples):
    mgs.sample()
    plt.scatter(mics.mpos[0], mics.mpos[1],marker="o",s=1,color="gray")
plt.scatter(mgs.mpos_init[0], mgs.mpos_init[1],marker="x",s=10,label="true positions")
plt.legend()    
plt.show()
