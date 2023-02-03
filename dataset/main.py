import argparse
from datetime import datetime
from os import path


def _create_filename(dataset, split, size, startsample, format):
    dirpath = path.dirname(path.abspath(__file__)) 
    metadata = dataset.get_dataset_metadata()
    name = f"{split}_"
    name += f"{startsample}-{startsample+size-1}_"
    name += "_".join([k+"-"+str(v) for k,v in metadata.items()]) + "_"
    name += f"{datetime.now().strftime('%d-%b-%Y')}"
    name += f".{format}"
    return path.join(dirpath,name)
    
def main(
    dataset,
    split,
    size,
    features,
    f,
    num,
    startsample,
    tasks,
    head,
    name,
    format,
    log):
    
    if dataset == "dataset1":
        from acoupipe.datasets.dataset1 import Dataset1 as Dataset
    elif dataset == "dataset2":
        from acoupipe.datasets.dataset2 import Dataset2 as Dataset

    # create dataset
    dataset = Dataset(features=features, f=f, num=num)

    # create filename if not given
    if name is None:
        name = _create_filename(dataset,split,size,startsample,format)
    print(f"Creating dataset '{name}'...")

    # save file
    if format == "h5":
        dataset.save_h5(split=split, tasks=tasks, startsample=startsample, 
                        address=head, size = size, name=name, log=log)    
    elif format == "tfrecord":
        dataset.save_tfrecord(split=split, tasks=tasks, startsample=startsample, 
                        address=head, size = size, name=name,log=log)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"],
                        help="Which dataset to compute. Default is 'dataset1'")
    parser.add_argument("--name", type=str, default=None,
                        help="filename of simulated data. If 'None' a filename is given and the file is stored under './datasets'")
    parser.add_argument("--format", type=str, default="h5", choices=["tfrecord", "h5"],
                        help="Desired file format to store the datasets. Defaults to '.h5' format")
    parser.add_argument("--features", nargs="+", default=["csm"], choices=["sourcemap", "csmtriu", "csm", "eigmode"],
                        help="Features included in the dataset. Default is the cross-spectral matrix 'csm'")
    parser.add_argument("--f", type=float, nargs="+", default=None,
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument("--num", type=int, default=0,
                        help="bandwidth of the considered frequencies. Default is single frequency line(s)")
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation", "test"],
                        help="Which dataset split to compute ('training' or 'validation' or 'test')")
    parser.add_argument("--size", type=int, required=True,
                        help="Total number of samples to simulate")
    parser.add_argument("--startsample", type=int, default=1,
                        help="Start simulation at a specific sample of the dataset. Default: 1")                    
    parser.add_argument("--tasks", type=int, default=1,
                        help="Number of asynchronous tasks. Defaults to '1' (non-distributed)")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.") 
    parser.add_argument("--log", action="store_true",
                        help="Whether to log timing statistics to file. Only for internal use.")                          

    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)

  
