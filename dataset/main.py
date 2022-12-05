import argparse
import yaml
from datetime import datetime
from os import path

def _create_filename(split,size,features,startsample,f,num,config,format):
    dirpath = path.dirname(path.abspath(__file__))
    if f:
        freq_str=",".join(str(int(i))+"Hz" for i in f)
    else:
        freq_str=f"fullfreq"    
    name = f"{split}_"
    name += f"{startsample}-{startsample+size-1}_"
    name += ",".join(features)+"_"
    name += f"{config['min_nsources']}-{config['max_nsources']}srcs_"
    name += f"{freq_str}-num{num}_"
    name += f"{config['version']}_"
    name += f"{datetime.now().strftime('%d-%b-%Y')}"
    name += f".{format}"
    return path.join(dirpath,"datasets",name)
    
def _get_config(default_config, config):
    with open(config, 'r') as c:
        new_config = yaml.safe_load(c)    
    default_config.update(new_config)
    return default_config

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
    config,
    log):
    
    if dataset == 'dataset1':
        from acoupipe.dataset1 import config1 as config, Dataset1 as Dataset
    
    # update config if necessary
    if config:
        config = _get_config(default_config = ds.config, config=config)
    else:
        config = ds.config

    # create dataset
    dataset = ds.Dataset(
        split = split,
        size = size,
        features = features,
        startsample = startsample,
        f = f,
        num = num,
        tasks = tasks,
        head = head,
        config = config,
    )

    # create filename if not given
    if name is None:
        name = _create_filename(split,size,features,startsample,f,num,config,format)

    # save file
    if format == 'h5':
        dataset.save_h5(name=name,log=log)    
    elif format == 'tfrecord':
        dataset.save_tfrecord(name=name,log=log)

 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dataset1", choices=["dataset1"],
                        help="Which dataset to compute")
    parser.add_argument('--split', type=str, default="training", choices=["training", "validation", "test"],
                        help="Which dataset split to compute ('training' or 'validation' or 'test')")
    parser.add_argument('--features', nargs="+", default=["csm"], choices=["sourcemap", "csmtriu", "csm", "ref_cleansc"],
                        help="Features included in the dataset. Default is the cross-spectral matrix 'csm'")
    parser.add_argument('--f', type=float, nargs="+", default=None,
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument('--num', type=int, default=0,
                        help="bandwidth of the considered frequencies. Default is single frequency line(s)")
    parser.add_argument('--size', type=int, required=True,
                        help="Total number of samples to simulate")
    parser.add_argument('--startsample', type=int, default=1,
                        help="Start simulation at a specific sample of the dataset. Default: 1")                    
    parser.add_argument('--name', type=str, default=None,
                        help="filename of simulated data. If 'None' a filename is given and the file is stored under './datasets'")
    parser.add_argument('--format', type=str, default="h5", choices=["tfrecord", "h5"],
                        help="Desired file format to store the datasets. Defaults to '.h5' format")
    parser.add_argument('--tasks', type=int, default=1,
                        help="Number of asynchronous tasks. Defaults to '1' (non-distributed)")
    parser.add_argument('--head', type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.") 
    parser.add_argument('--config', type=str, default=None,
                        help="Optional config.yml file specifying underlying parameters.")    
    parser.add_argument('--log', action="store_true",
                        help="Whether to log timing statistics to file. Only for internal use.")                          

    # parse and run main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)

  