from os import path
import acoular
from pylab import figure, plot, axis, imshow, colorbar, show
from acoupipe import PlanarSourceMapEvaluator
import numpy as np

acoular.config.global_caching = "none"

micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
mg = acoular.MicGeom( from_file=micgeofile )

wn1 = acoular.WNoiseGenerator(sample_freq=51200,seed=1,numsamples=51200)
wn2 = acoular.WNoiseGenerator(sample_freq=51200,seed=2,numsamples=51200,rms=.5)
wn3 = acoular.WNoiseGenerator(sample_freq=51200,seed=3,numsamples=51200,rms=.75)

ps1 = acoular.PointSource(signal=wn1,mics=mg, loc=(0.11,0.15,.5))
ps2 = acoular.PointSource(signal=wn2,mics=mg, loc=(0.1,0.15,.5))
ps3 = acoular.PointSource(signal=wn3,mics=mg, loc=(-0.1,-0.15,.5))

ts = acoular.SourceMixer( sources=[ps1,ps2,ps3])

#ts = acoular.PointSource(signal=wn1,mics=mg, loc=(-0.15,0.0,.2))

ps = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=.5, \
increment=0.01 )
st = acoular.SteeringVector( grid = rg, mics=mg,ref=mg.mpos[:,8] )
bb = acoular.BeamformerCleansc( freq_data=ps, steer=st )
ps_ref = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
ps_ref.time_data = acoular.MaskedTimeInOut(source=ts,invalid_channels=[_ for _ in range(64) if not _  == 8]) # masking other channels than the reference channel

ps_ref.time_data=ps1
p2_ps1 = ps_ref.csm[:,0,0] # power spectra
ps_ref.time_data=ps2
p2_ps2 = ps_ref.csm[:,0,0] # power spectra
ps_ref.time_data=ps3
p2_ps3 = ps_ref.csm[:,0,0] # power spectra

# ref values
p2_ref = np.real(np.array([p2_ps1,p2_ps2,p2_ps3]).T)
loc_ref = np.array([ps1.loc,ps2.loc,ps3.loc])
sourcemaps = np.array([bb.synthetic( f, 0 ) for f in ps.fftfreq()])

# p2_ps1 = ps_ref.csm[:,0,0] # power spectra
# p2_ref = np.real(np.array([p2_ps1]).T)
# loc_ref = np.array([ts.loc])
# sourcemaps = np.array([bb.synthetic( f, 0 ) for f in ps.fftfreq()])

figure()
Lm = acoular.L_p(sourcemaps[50]).T
im = imshow(Lm,vmax=Lm.max(), vmin=Lm.max()-20,extent=rg.extend())
colorbar()

se = PlanarSourceMapEvaluator(sourcemap=sourcemaps, grid=rg, target_loc=loc_ref, target_p2=p2_ref, r=0.05)
se.get_overall_level_error()
res = se._integrate_targets()
res = se.get_specific_level_error()
res = se.get_inverse_level_error()


#se = PlanarSourceMapEvaluator(beamformer=bb, target_loc=loc_ref, target_p2=p2_ref, r=0.05)
# print("overall level error: ", se.overall_level_error(FREQ,NUM))
# print("inverse level error: ", se.inverse_level_error(FREQ,NUM))
# print("specific level error: ", se.specific_level_error(FREQ,NUM))