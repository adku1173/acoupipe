import argparse
import socket
from datetime import datetime
from os import path
from time import time

import pandas as pd


def _create_filename():
    dirpath = path.dirname(path.abspath(__file__)) 
    timestamp = datetime.now().strftime("%d-%b-%Y")
    name = f"throughput_{timestamp}.pkl"
    return path.join(dirpath,"results",name)
    
def main(
    dataset,
    size,
    features,
    f,
    tasks,
    head,
    log):
    
    if dataset == "dataset1":
        from acoupipe.datasets.dataset1 import Dataset1 as Dataset
    elif dataset == "dataset2":
        from acoupipe.datasets.dataset2 import Dataset2 as Dataset

    # create dataset
    dataset = Dataset(features=features, f=f)

    size += 1 # we add one sample to compensate the ray startup time
    gen = dataset.generate(split="training", tasks=tasks, size=size, log=log, address=head, progress_bar=False)
    next(gen)

    t1 = time()
    for _d in gen:
        pass 
    t = time() - t1
    data = [[Dataset.__name__, features[0], (size-1)/t, size-1, tasks, t, 
                    socket.gethostname(), head, dataset.get_dataset_metadata()["version"]]]

    filename = _create_filename()
    if path.exists(filename):
        df = pd.read_pickle(filename)
        # add pandas row
        df.loc[len(df)] = data[0]
        df.to_pickle(filename)
    else:
        df = pd.DataFrame(columns=["dataset","feature", "throughput", "size", "tasks", "time", "hostname", "head", "version"], 
            data=data)
        df.to_pickle(filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"],
                        help="Which dataset to compute. Default is 'dataset1'")
    parser.add_argument("--features", nargs="+", default=["csm"], choices=["sourcemap", "csmtriu", "csm", "eigmode"],
                        help="Features included in the dataset. Default is the cross-spectral matrix 'csm'")
    parser.add_argument("--f", type=float, nargs="+", default=None,
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument("--size", type=int, default=2,
                        help="Total number of samples to simulate")
    parser.add_argument("--tasks", type=int, default=1,
                        help="Number of asynchronous tasks. Defaults to '1' (non-distributed)")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.") 
    parser.add_argument("--log", action="store_true",
                        help="Whether to log timing statistics to file. Only for internal use.")                          

    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)

  
