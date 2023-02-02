from acoupipe.config import TF_FLAG
from acoupipe.writer import WriteH5Dataset

if TF_FLAG:
    from acoupipe.writer import WriteTFRecord
import logging
from datetime import datetime
from os import path
from os.path import join
from warnings import warn

from acoular import config
from numpy import concatenate, imag, newaxis, real, searchsorted


def _handle_cache(cache_bf,cache_csm,cache_dir):
    # Caching 
    if cache_bf or cache_csm:
        if not cache_dir:
            raise ValueError("Please specify a cache_dir via the --cache_dir flag!")
        config.cache_dir = path.join(cache_dir,"cache") # set up cache file dir
        print("cache file directory at: ",config.cache_dir)
        cache_dir = config.cache_dir
    else: # nothing is cached
        cache_dir = None
    return cache_dir

def _handle_log(fname):
    logging.basicConfig(level=logging.INFO) # root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    pipeline_log = logging.FileHandler(fname,mode="w") # log everything to file
    pipeline_log.setFormatter(logging.Formatter(
        "%(process)d-%(levelname)s-%(asctime)s.%(msecs)02d-%(message)s", datefmt="%Y-%m-%d,%H:%M:%S"))
    logger.addHandler(pipeline_log) # attach handler to the root logger

def get_frequency_index_range(freq,f,num):
    """Returns the left and right indices that define the frequency range to integrate over.

    Parameters
    ----------
    freq : numpy.array
        frequency vector (can be determined by evaluating `freqdata()` method at a `acoular.PowerSpectra` instance)
    f : float
        the frequency (or center frequency) of interest
    num : int
        the frequency band (0: single frequency line, 1: octave band, 3: third octave band)

    Returns
    -------
    tuple
        left and right index that belongs to the frequency of interest
    """
    if num == 0:
        # single frequency line
        ind = searchsorted(freq, f)
        if ind >= len(freq):
            warn("Queried frequency (%g Hz) not in resolved "
                            "frequency range. Returning zeros." % f, 
                            Warning, stacklevel = 2)
            ind = None
        else:
            if freq[ind] != f:
                warn("Queried frequency (%g Hz) not in set of "
                        "discrete FFT sample frequencies. "
                        "Using frequency %g Hz instead." % (f,freq[ind]), 
                        Warning, stacklevel = 2)
        return (ind,ind+1)
    else:
        # fractional octave band
        if isinstance(num,list):
            f1=num[0]
            f2=num[-1]
        else:
            f1 = f*2.**(-0.5/num)
            f2 = f*2.**(+0.5/num)
        ind1 = searchsorted(freq, f1)
        ind2 = searchsorted(freq, f2)
        if ind1 == ind2:
            warn("Queried frequency band (%g to %g Hz) does not "
                    "include any discrete FFT sample frequencies. "
                    "Returning zeros." % (f1,f2), 
                    Warning, stacklevel = 2)
        return (ind1,ind2) 


def set_pipeline_seeds(pipeline,startsample,size,dataset="training"):
    """Creates the random seed list for each of the sampler objects that is held by the pipeline object.

    Parameters
    ----------
    pipeline : instance of class BasePipeline
        the pipeline object holding the sampler classes
    startsample : int
        start sample to be calculated by the pipeline        
    size : int
        number of samples to be yielded by the pipeline
    dataset : str, optional
        the data set type, by default "training". Choose from ["training","validation"]
    """
    startindex = startsample-1 # index starts at 0
    if dataset=="training":
        off = 0
    elif dataset=="validation":
        off = int(1e9) # a general offset to ensure that validation and training seeds never match (max seed is 2**32) 
    elif dataset == "test":
        off = int(2e9)
    soff = int(1e7) # offset to ensure that seeds of sampler object doesn't match
    pipeline.random_seeds = {i : range(off+(i*soff)+startindex, off+(i*soff)+size+startindex) for i in list(pipeline.sampler.keys())}
    
        

def set_filename(writer,path=".",*args):
    """Sets the filename of the dataset.

    Parameters
    ----------
    writer : instance of class BaseWriteDataset
        the writer object holding the filename
    path : str, optional
        the path to the dataset, current directory by default
    *args : str
        concatenated strings to be used as the filename
    """
    name = f"{args[0]}"
    for arg in args[1:]:
        name += f"_{arg}"
    name += f"_{datetime.now().strftime('%d-%b-%Y')}"   
    if isinstance(writer,WriteH5Dataset):
        name += ".h5"
    if TF_FLAG:
        if isinstance(writer,WriteTFRecord):
            name += ".tfrecord"
    writer.name=join(path,name)


def complex_to_real(func):
    def complex_to_real_wrapper(*args,**kwargs):
        a = func(*args,**kwargs)
        return concatenate(
            [real(a)[...,newaxis],
            imag(a)[...,newaxis]],axis=-1)
    return complex_to_real_wrapper
