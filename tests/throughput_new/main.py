import argparse
import logging
import socket
from time import time

import numba

from acoupipe.new_datasets.synthetic import DatasetSynthetic as NewDatasetSynthetic


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

def main(size,head,loglevel):
    logging.basicConfig(level=loglevel)
    logging.info(f"Running throughput test on {socket.gethostname()}")
    logging.info(f"Numba threading layer: {numba.config.THREADING_LAYER}")


    logging.info(f"Starting ray cluster with head node {head}")
    # ray.init(head)

    # Test NewDatasetSynthetic (Ray Dataset)
    newds = NewDatasetSynthetic(mic_pos_noise=False, signal_length=5.)
    newds.signal_model.dtype = "float16"

    # Measure execution time
    ray_ds = newds.get_ray_dataset(features=["idx", "time_data"], size=size)
    for _ in ray_ds.iter_rows():
        pass

    # log stats
    logging.info("Ray Dataset stats:")
    logging.info(ray_ds.stats())



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500,
                        help="Total number of samples to simulate")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)


