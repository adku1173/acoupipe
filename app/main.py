import argparse
from datetime import datetime
from pathlib import Path


def _create_filename(dataset, split, size, start_idx, format):
    dirpath = Path(__file__).parent.absolute() / "datasets"
    if not dirpath.exists():
        dirpath.mkdir()
    name = f"{dataset}_{split}_"
    name += f"{start_idx+1}-{start_idx+size}_"
    name += f"{datetime.now().strftime('%d-%b-%Y')}"
    name += f".{format}"
    return dirpath / name

def main(
    dataset,
    mode,
    split,
    size,
    features,
    f,
    num,
    start_idx,
    tasks,
    head,
    name,
    format,
    log):

    if log:
        import logging
        logging.basicConfig(level=logging.INFO)

    if dataset == "DatasetSynthetic":
        from acoupipe.datasets.synthetic import DatasetSynthetic as Dataset
    elif dataset == "DatasetMIRACLE":
        from acoupipe.datasets.experimental import DatasetMIRACLE as Dataset

    # create filename if not given
    if name is None:
        name = _create_filename(dataset,split,size,start_idx,format)
    print(f"Creating dataset '{name}'...")

    if tasks > 1:
        import ray
        ray.init(address=head, log_to_driver=log)

    # save file
    if format == "h5":
        Dataset(tasks=tasks,mode=mode).save_h5(features=features, f=f, num=num,
                        split=split, start_idx=start_idx,
                        size = size, name=name)
    elif format == "tfrecord":
        Dataset(tasks=tasks,mode=mode).save_tfrecord(
                        features=features, f=f, num=num,
                        split=split, start_idx=start_idx,
                        size = size, name=name)

if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DatasetSynthetic", choices=["DatasetSynthetic", "DatasetMIRACLE"],
                        help="Which dataset to compute. Default is 'DatasetSynthetic'")
    parser.add_argument("--name", type=str, default=None,
                        help="filename of simulated data. If 'None' a filename is given and the file is stored under './datasets'")
    parser.add_argument("--features", nargs="+", default=[], choices=[
                                    "time_data","csm","csmtriu","sourcemap","eigmode", "spectrogram", "loc","source_strength_analytic",
                                    "source_strength_estimated", "noise_strength_analytic","noise_strength_estimated"],
                        help="Features included in the dataset. Default is the cross-spectral matrix 'csm'")
    parser.add_argument("--mode", type=str, default="welch", choices=["welch", "wishart", "analytic"],
                        help="Calculation mode of the underlying Cross-spectral matrix. Default is 'welch'")
    parser.add_argument("--format", type=str, default="h5", choices=["tfrecord", "h5"],
                        help="Desired file format to store the datasets. Defaults to '.h5' format")
    parser.add_argument("--f", type=float, nargs="+", default=None,
                        help="frequency or frequencies included by the features and labels. Default is 'None' (all frequencies included)")
    parser.add_argument("--num", type=int, default=0,
                        help="bandwidth of the considered frequencies. Default is single frequency line(s)")
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation", "test"],
                        help="Which dataset split to compute ('training' or 'validation' or 'test')")
    parser.add_argument("--size", type=int, required=True,
                        help="Total number of samples to simulate")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start simulation at a specific sample of the dataset. Default: 0")
    parser.add_argument("--tasks", type=int, default=1,
                        help="Number of asynchronous tasks. Defaults to '1' (non-distributed)")
    parser.add_argument("--head", type=str, default=None,
                        help="IP address of the head node in the ray cluster. Only necessary when running in distributed mode.")
    parser.add_argument("--log", action="store_true",
                        help="Whether to log timing statistics. Only for internal use.")

    # parse and execute main function
    kwargs = vars(parser.parse_args())
    main(**kwargs)


