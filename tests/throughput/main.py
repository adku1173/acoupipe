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
    
def main(task_list,size,f,head,log):
    filename = _create_filename()

    for dataset in ["dataset1", "dataset2"]:

        if dataset == "dataset1":
            from acoupipe.datasets.dataset1 import Dataset1 as Dataset
        elif dataset == "dataset2":
            from acoupipe.datasets.dataset2 import Dataset2 as Dataset
        for tasks in task_list:
            for feature in ["sourcemap", "csmtriu", "csm", "eigmode"]:

                initial_size = size*tasks # scale the initial size with the number of tasks

                # create dataset
                dataset = Dataset(features=[feature], f=f)

                gen = dataset.generate(split="training", tasks=tasks, size=initial_size+1, log=log, address=head, progress_bar=False)
                next(gen)

                t1 = time()
                for _d in gen:
                    pass 
                t = time() - t1
                print(feature,t,tasks)
                data = [[Dataset.__name__, feature, initial_size/t, initial_size, tasks, t, 
                                socket.gethostname(), head, dataset.get_dataset_metadata()["version"]]]

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
    parser.add_argument("--task_list", type=int, nargs="+", default=[1,2,4,8,16,32],)
    parser.add_argument("--f", type=float, nargs="+", default=None,
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument("--size", type=int, default=500,
                        help="Total number of samples to simulate")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.") 
    parser.add_argument("--log", action="store_true",
                        help="Whether to log timing statistics to file. Only for internal use.")                          

    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)

  
