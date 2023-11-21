import argparse
import logging
import shutil
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from time import time

import numba
import pandas as pd
import ray


def get_dataset(dataset,**kwargs):
    if dataset == "DatasetSynthetic1":
        from acoupipe.datasets.synthetic import DatasetSynthetic1 as Dataset
        kwargs.pop("srir_dir")
        return Dataset(**kwargs)
    elif dataset == "DatasetMIRACLE":
        from acoupipe.datasets.experimental import DatasetMIRACLE as Dataset
        return Dataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")

def get_method(method):
    if method == "generate":
        return time_generate
    elif method == "tfdataset":
        return time_tfdataset
    elif method == "save_h5":
        return time_save_h5
    elif method == "save_tfrecord":
        return time_save_tfrecord
    else:
        raise ValueError(f"Method {method} not recognized.")

def create_filename():
    dirpath = Path(__file__).parent.absolute()
    timestamp = datetime.now().strftime("%d-%b-%Y")
    name = f"throughput_{socket.gethostname()}_{timestamp}.pkl"
    return dirpath / "results" / name

def time_generate(dataset, **kwargs):
    gen = dataset.generate(split="training", progress_bar=False, **kwargs)
    t1 = time()
    next(gen)
    t_startup = time() - t1
    t1 = time()
    for _ in gen:
        pass
    t = time() - t1
    return t_startup, t

def time_tfdataset(dataset, **kwargs):
    gen = iter(dataset.get_tf_dataset(split="training", progress_bar=False, **kwargs))
    t1 = time()
    next(gen)
    t_startup = time() - t1
    t1 = time()
    for _ in gen:
        pass
    t = time() - t1
    return t_startup, t

def time_save_h5(dataset, **kwargs):
    dir = Path(tempfile.mkdtemp())
    name = dir / "dataset.h5"
    t1 = time()
    dataset.save_h5(split="training", name=name, progress_bar=False, **kwargs)
    t = time() - t1
    shutil.rmtree(dir)
    return None, t

def time_save_tfrecord(dataset, **kwargs):
    dir = Path(tempfile.mkdtemp())
    name = dir / "dataset.tfrecord"
    t1 = time()
    dataset.save_tfrecord(split="training", name=name, progress_bar=False, **kwargs)
    t = time() - t1
    shutil.rmtree(dir)
    return None, t

def main(datasets,methods,modes,task_list,features,size,freqs,head,srirdir,loglevel):
    logging.basicConfig(level=loglevel)
    logging.info(f"Running throughput test on {socket.gethostname()}")
    logging.info(f"Numba threading layer: {numba.config.THREADING_LAYER}")
    filename = create_filename()
    for dataset_name in datasets:
        for method_name in methods:
            for mode in modes:
                for tasks in task_list:
                    for feature in features:
                        for f in freqs:

                            if (mode != "welch" and "time_data" in feature) or (mode != "welch" and "spectrogram" in feature):
                                continue

                            logging.info("---------------------------------------------------")
                            logging.info(f"Running {dataset_name} {method_name} {mode} {tasks} {feature} {f}")
                            logging.info("---------------------------------------------------")

                            if tasks > 1:
                                logging.info(f"Starting ray cluster with head node {head}")
                                ray.init(head, log_to_driver=False)

                            # create dataset
                            if feature == "None":
                                feat = []
                            else:
                                feat=[feature]

                            dataset = get_dataset(dataset_name, mode=mode, tasks=tasks, srir_dir=srirdir)
                            method = get_method(method_name)
                            # run method
                            initial_size = size*tasks # scale the initial size with the number of tasks
                            t_startup, t = method(dataset, features=feat, f=f, size=initial_size+1)
                            # collect data and save to dataframe
                            throughput = initial_size/t
                            logging.info(f"Startup time: {t_startup} s")
                            logging.info(f"Throughput: {throughput} samples/s")
                            data = [[dataset_name, method_name, mode, feature, f, initial_size, throughput, tasks, t_startup, t, head]]
                            if filename.exists():
                                df = pd.read_pickle(filename)
                                # add pandas row
                                df.loc[len(df)] = data[0]
                                df.to_pickle(filename)
                            else:
                                df = pd.DataFrame(
                                    columns=[
                                        "dataset","method", "mode", "feature", "f", "size", "throughput","tasks", "startup_time", "time", "head"],
                                    data=data)
                                df.to_pickle(filename)
                            if tasks > 1:
                                logging.info(f"Shutting down ray cluster with head node {head}")
                                ray.shutdown()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=["DatasetSynthetic1"], choices=["DatasetSynthetic1", "DatasetMIRACLE"])
    parser.add_argument("--task_list", type=int, nargs="+", default=[1])
    parser.add_argument("--features", type=str, nargs="+", default=["spectrogram","sourcemap","csmtriu","csm","eigmode","time_data"])
    parser.add_argument("--methods", type=str, nargs="+", default=["generate"])
    parser.add_argument("--freqs", type=float, nargs="+", default=[4000, None],
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument("--size", type=int, default=100,
                        help="Total number of samples to simulate")
    parser.add_argument("--modes", type=str, default=["welch", "wishart", "analytic"], nargs="+")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.")
    parser.add_argument("--srirdir", type=str, default=None,
                        help="Path to the SRIR directory (only necessary for DatasetMIRACLE)")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)


