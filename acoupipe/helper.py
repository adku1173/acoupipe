from acoupipe import WriteH5Dataset, WriteTFRecord
from datetime import datetime
from os.path import join


def set_pipeline_seeds(pipeline,startsample,numsamples,dataset="training"):
    """creates the random seed list for each of the sampler objects that is held by the 
    pipeline object.

    Parameters
    ----------
    pipeline : instance of class BasePipeline
        the pipeline object holding the sampler classes
    startsample : int
        start sample to be calculated by the pipeline        
    numsamples : int
        number of samples to be yielded by the pipeline
    dataset : str, optional
        the data set type, by default "training". Choose from ["training","validation"]
    """
    startindex = startsample-1 # index starts at 0
    if dataset=="training":
        off = 0
    elif dataset=="validation":
        off = int(1e9) # a general offset to ensure that validation and training seeds never match (max seed is 2*32) 
    soff = int(1e7) # offset to ensure that seeds of sampler object doesn't match
    pipeline.random_seeds = [range(off+(i*soff)+startindex, off+(i*soff)+numsamples+startindex) for i in range(len(pipeline.sampler))]

def set_filename(writer,path='.',*args):
    """sets the filename of the data set.

    Parameters
    ----------
    writer : instance of class BaseWriteDataset
    """
    name = f"{args[0]}"
    for arg in args[1:]:
        name += f"_{arg}"
    name += f"_{datetime.now().strftime('%d-%b-%Y')}"
    if isinstance(writer,WriteTFRecord):
        name += ".tfrecord"
    elif isinstance(writer,WriteH5Dataset):
        name += ".h5"
    writer.name=join(path,name)
