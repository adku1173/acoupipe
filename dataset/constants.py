
from os import path
dirpath = path.dirname(path.abspath(__file__))

ds1_constants = {
    'AP' : 1.,
    'VERSION':"ds1-v001", # data set version
    'C' : 343., # speed of sound
    'HE' : 40, # Helmholtz number (defines the sampling frequency) 
    'SFREQ' : 40*343./1., # /ap with ap:1.0
    'BLOCKSIZE' : 128, # block size used for FFT 
    'OVERLAP' : '50%',
    'WINDOW' : 'Hanning',
    'SIGLENGTH':5, # length of the simulated signal
    'MFILE' : path.join(dirpath,"tub_vogel64_ap1.xml"), # Microphone Geometry
    'REF_MIC' : 63, # index of the reference microphone 
}