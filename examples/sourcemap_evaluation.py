
import os

import acoular
import matplotlib.pyplot as plt
import numpy as np
from acoular import BeamformerCleansc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from acoupipe.datasets.dataset2 import DEFAULT_BEAMFORMER, DEFAULT_GRID, Dataset2
from acoupipe.evaluate import SourceMapEvaluator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # change tensorflow log level for doc purposes


beamformer = BeamformerCleansc(freq_data=DEFAULT_BEAMFORMER.freq_data)

# sourcemap dataset
dataset = Dataset2(max_nsources=3, min_nsources=3, f=1000, beamformer=beamformer, features=["sourcemap"])    

data = next(dataset.generate(split="training", size=1))

sourcemap = data["sourcemap"]
loc_ref = data["loc"]
print(loc_ref)
p2_ref = data["p2"]

# prepare to calculate metrics
se = SourceMapEvaluator(sourcemap=sourcemap, grid=DEFAULT_GRID, 
                                target_loc=loc_ref[:2], target_pow=p2_ref, r=0.06,
                                variable_sector_radii=True)

#%% plot sourcemaps with variable and fixed sector sizes
Lm = acoular.L_p(sourcemap[0])

fig = plt.figure(1,(8,4))
for i in range(1,3):
    # calculate metrics
    specific_level_error = se.get_specific_level_error()[0]
    inverse_level_error = se.get_inverse_level_error()[0]
    overall_level_error = se.get_overall_level_error()[0]
    # subplot results
    ax = fig.add_subplot(1,2,i)
    if i == 1:
        ax.set_title("variable sector size")
    if i == 2:
        ax.set_title("fixed sector size")
    im = ax.imshow(Lm.T,vmax=Lm.max(),vmin=Lm.max()-20,cmap="hot_r",extent=DEFAULT_GRID.extend(),origin="lower")
    #plot true positions
    for j in range(loc_ref.shape[-1]):
        l = loc_ref[:,j]
        ax.add_patch(plt.Circle((l[0], l[1]), se.sector_radii[j], color="black",fill=False))
        ax.annotate(xycoords="data",xy=(l[0],l[0]),xytext=(l[0]+0.02,l[1]),text=r"$L_{p,s}$="+f"{np.round(specific_level_error[j],1)} dB")
    ax.text(0.2, 0.9,r"$L_{p,o}$="+f"{np.round(overall_level_error,1)} dB",horizontalalignment="center", verticalalignment="center", transform = ax.transAxes)
    ax.text(0.2, 0.8,r"$L_{p,i}$="+f"{np.round(inverse_level_error,1)} dB",horizontalalignment="center", verticalalignment="center", transform = ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-.2,.2)
    ax.set_ylim(-.2,.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        position="right", 
        size="5%", 
        pad=0.0,)
    cax.tick_params(direction="in")
    fig.colorbar(im,cax)
plt.tight_layout()
plt.show()

# %% provide an example map for a gridless result

# rng = np.random.RandomState(1)
# estimated_loc = loc_ref + rng.random(loc_ref.shape)*1e-2
# estimated_pow = p2_ref + rng.random(p2_ref.shape)*1e-1
# ge = GridlessEvaluator(
#     target_loc=loc_ref, 
#     target_pow=p2_ref, 
#     estimated_loc=estimated_loc,
#     estimated_pow=estimated_pow,
#     r=0.06)


# vmax = acoular.L_p(estimated_pow.max())
# vmin = vmax-20
# norm = matplotlib.colors.Normalize(vmax=vmax,vmin=vmin)
# cmap = plt.cm.get_cmap("hot_r")

# fig = plt.figure(2,(8,4))
# se.variable_sector_radii = True
# for i in range(1,3):
#     # calculate metrics
#     specific_level_error = ge.get_specific_level_error()[0]
#     inverse_level_error = ge.get_inverse_level_error()[0]
#     overall_level_error = ge.get_overall_level_error()[0]
#     # subplot results
#     ax = fig.add_subplot(1,2,i)
#     if i == 1:
#         ax.set_title("variable sector size")
#     if i == 2:
#         ax.set_title("fixed sector size")
#     im = ax.imshow(np.zeros(DEFAULT_GRID.shape),vmax=Lm.max(),vmin=Lm.max()-20,cmap="hot_r",extent=DEFAULT_GRID.extend(),origin="lower")
#     # plot esimated positons
#     for j,l in enumerate(estimated_loc):
#         ax.plot(l[0], l[1],"o",color=cmap(norm(acoular.L_p(estimated_pow[0][j]))))
#     #plot circles around true positions
#     for j,l in enumerate(loc_ref):
#         ax.add_patch(plt.Circle((l[0], l[1]), se.sector_radii[j], color="black",fill=False))
#         ax.annotate(xycoords="data",xy=(l[0],l[0]),xytext=(l[0]+0.02,l[1]),text=r"$L_{p,s}$="+f"{np.round(specific_level_error[j],1)} dB")
#     ax.text(0.2, 0.9,r"$L_{p,o}$="+f"{np.round(overall_level_error,1)} dB",horizontalalignment="center", verticalalignment="center", transform = ax.transAxes)
#     ax.text(0.2, 0.8,r"$L_{p,i}$="+f"{np.round(inverse_level_error,1)} dB",horizontalalignment="center", verticalalignment="center", transform = ax.transAxes)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_xlim(-.2,.2)
#     ax.set_ylim(-.2,.2)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes(
#         position="right", 
#         size="5%", 
#         pad=0.0,)
#     cax.tick_params(direction="in")
#     fig.colorbar(im,cax)
#     se.variable_sector_radii = False
# plt.tight_layout()
# plt.show()
# # %%
