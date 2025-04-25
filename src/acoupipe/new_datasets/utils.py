

def add_split(data, split):
    data["split"] = split
    return data

def get_seed(idx, sampler_id, split="training"):
    """Create the random seed list for each of the sampler objects that is held by the pipeline object.

    Parameters
    ----------
    pipeline : instance of class BasePipeline
        the pipeline object holding the sampler classes
    start_idx : int
        start index to be calculated by the pipeline
    size : int
        number of samples to be yielded by the pipeline
    split : str, optional
        the data set type, by default "training". Choose from ["training","validation"]
    """
    if split == "training":
        off = 0
    elif split == "validation":
        # we assume that the training set will never be larger than 1e12 samples
        off = int(1e12)  # a general offset to ensure that validation and training seeds never match
    elif split == "test":
        off = int(1e21)
    soff = off + (sampler_id * int(1e7))  # offset to ensure that seeds of sampler object doesn't match
    return soff + idx
