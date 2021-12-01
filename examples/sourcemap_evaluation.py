from os import path
import acoular
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from acoupipe import PlanarSourceMapEvaluator, GridlessEvaluator, get_frequency_index_range
import numpy as np

micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
mg = acoular.MicGeom( from_file=micgeofile )

# create noise sources
wn1 = acoular.WNoiseGenerator(sample_freq=51200,seed=1,numsamples=51200*5)
wn2 = acoular.WNoiseGenerator(sample_freq=51200,seed=2,numsamples=51200*5,rms=.5)
wn3 = acoular.WNoiseGenerator(sample_freq=51200,seed=3,numsamples=51200*5,rms=.75)

# create spatially distributed monopoles
ps1 = acoular.PointSource(signal=wn1,mics=mg, loc=(0.0,0,.5))
ps2 = acoular.PointSource(signal=wn2,mics=mg, loc=(0.,0.075,.5))
ps3 = acoular.PointSource(signal=wn3,mics=mg, loc=(-0.1,-0.15,.5))
ts = acoular.SourceMixer( sources=[ps1,ps2,ps3])
cache = acoular.TimeCache(source=ts)

# process microphone array signals
ps = acoular.PowerSpectra( time_data=cache, block_size=1024, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=.5, \
increment=0.01 )
st = acoular.SteeringVector( grid = rg, mics=mg,ref=mg.mpos[:,8] )
bb = acoular.BeamformerCleansc( freq_data=ps, steer=st )
# calculate CLEAN-SC map at 8 kHz third octave
sourcemap = bb.synthetic( 8000, 3 )

# get the reference auto-power values at the reference microphone for each point source and for the 
# full 8 kHz frequency band
ps_ref = acoular.PowerSpectra( time_data=ts, block_size=1024, window='Hanning' )
ps_ref.time_data = acoular.MaskedTimeInOut(source=ts,invalid_channels=[_ for _ in range(64) if not _  == 8]) # masking other channels than the reference channel
# calculate the power spectrum for each source signal at reference position
# 1.: get the frequency indices that belong to the 8 kHz band
fidx_range = get_frequency_index_range(ps.fftfreq(),8000,3)
# 2.: calculate the auto-power spectrum for each source separately and sum over the frequency indices
# to yield the target third-octave power
ps_ref.time_data=ps1
p2_ps1 = ps_ref.csm[fidx_range[0]:fidx_range[1],0,0].sum() # power spectra
ps_ref.time_data=ps2
p2_ps2 = ps_ref.csm[fidx_range[0]:fidx_range[1],0,0].sum() # power spectra
ps_ref.time_data=ps3
p2_ps3 = ps_ref.csm[fidx_range[0]:fidx_range[1],0,0].sum() # power spectra
# ref values
p2_ref = np.real(np.array([p2_ps1,p2_ps2,p2_ps3])[np.newaxis,:])
loc_ref = np.array([ps1.loc,ps2.loc,ps3.loc])

# prepare to calculate metrics
se = PlanarSourceMapEvaluator(sourcemap=sourcemap[np.newaxis,:,:], grid=rg, target_loc=loc_ref, target_pow=p2_ref, r=0.06)

#%% plot sourcemaps with variable and fixed sector sizes
Lm = acoular.L_p(sourcemap).T

fig = plt.figure(1,(8,4))
se.variable_sector_radii = True
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
    im = ax.imshow(Lm,vmax=Lm.max(),vmin=Lm.max()-20,cmap="hot_r",extent=rg.extend(),origin='lower')
    #plot true positions
    for j,l in enumerate(loc_ref):
        ax.add_patch(plt.Circle((l[0], l[1]), se.sector_radii[j], color='black',fill=False))
        ax.annotate(xycoords='data',xy=(l[0],l[0]),xytext=(l[0]+0.02,l[1]),text=r"$L_{p,s}$="+f"{np.round(specific_level_error[j],1)} dB")
    ax.text(0.2, 0.9,r"$L_{p,o}$="+f"{np.round(overall_level_error,1)} dB",horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.2, 0.8,r"$L_{p,i}$="+f"{np.round(inverse_level_error,1)} dB",horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-.2,.2)
    ax.set_ylim(-.2,.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        position='right', 
        size='5%', 
        pad=0.0,)
    cax.tick_params(direction='in')
    fig.colorbar(im,cax)
    se.variable_sector_radii = False
plt.tight_layout()

# %% provide an example map for a gridless result

rng = np.random.RandomState(1)
estimated_loc = loc_ref + rng.random(loc_ref.shape)*1e-2
estimated_pow = p2_ref + rng.random(p2_ref.shape)*1e-1
ge = GridlessEvaluator(
    target_loc=loc_ref, 
    target_pow=p2_ref, 
    estimated_loc=estimated_loc,
    estimated_pow=estimated_pow,
    r=0.06)


vmax = acoular.L_p(estimated_pow.max())
vmin = vmax-20
norm = matplotlib.colors.Normalize(vmax=vmax,vmin=vmin)
cmap = plt.cm.get_cmap('hot_r')

fig = plt.figure(2,(8,4))
se.variable_sector_radii = True
for i in range(1,3):
    # calculate metrics
    specific_level_error = ge.get_specific_level_error()[0]
    inverse_level_error = ge.get_inverse_level_error()[0]
    overall_level_error = ge.get_overall_level_error()[0]
    # subplot results
    ax = fig.add_subplot(1,2,i)
    if i == 1:
        ax.set_title("variable sector size")
    if i == 2:
        ax.set_title("fixed sector size")
    im = ax.imshow(np.zeros(rg.shape),vmax=Lm.max(),vmin=Lm.max()-20,cmap="hot_r",extent=rg.extend(),origin='lower')
    # plot esimated positons
    for j,l in enumerate(estimated_loc):
        ax.plot(l[0], l[1],'o',color=cmap(norm(acoular.L_p(estimated_pow[0][j]))))
    #plot circles around true positions
    for j,l in enumerate(loc_ref):
        ax.add_patch(plt.Circle((l[0], l[1]), se.sector_radii[j], color='black',fill=False))
        ax.annotate(xycoords='data',xy=(l[0],l[0]),xytext=(l[0]+0.02,l[1]),text=r"$L_{p,s}$="+f"{np.round(specific_level_error[j],1)} dB")
    ax.text(0.2, 0.9,r"$L_{p,o}$="+f"{np.round(overall_level_error,1)} dB",horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.2, 0.8,r"$L_{p,i}$="+f"{np.round(inverse_level_error,1)} dB",horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-.2,.2)
    ax.set_ylim(-.2,.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        position='right', 
        size='5%', 
        pad=0.0,)
    cax.tick_params(direction='in')
    fig.colorbar(im,cax)
    se.variable_sector_radii = False
plt.tight_layout()
# %%
