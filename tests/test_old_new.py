import acoular as ac # noqa: I001
import argparse
import time
import tracemalloc

import ray

from acoupipe.datasets.synthetic import DatasetSynthetic
from acoupipe.new_datasets.synthetic import DatasetSynthetic as NewDatasetSynthetic
print(ac.__version__)

# Initialize Ray with increased object store memory (e.g., 4 GB)
context = ray.init(object_store_memory=4 * 1024 * 1024 * 1024)
print(context.dashboard_url)
print(ray.available_resources())

# Preserve ordering in Ray Datasets for reproducibility.
ctx = ray.data.DataContext.get_current()
ctx.execution_options.preserve_order = False
ctx.execution_options.resource_limits.object_store_memory=4 * 1024 * 1024 * 1024

def measure_performance_ray(dataset, dataset_name, size, start_idx, **kwargs):
    """
    Measure execution time and memory usage for a Ray-based dataset implementation.

    Args:
        dataset: The dataset instance to test.
        dataset_name: Name of the dataset implementation (for reporting).
        size: Number of samples to generate.
        start_idx: Starting index for the dataset.
        **kwargs: Additional arguments for dataset generation.

    Returns
    -------
        None
    """
    print(f"\nTesting {dataset_name} (Ray Dataset)...")

    # Start memory tracing
    tracemalloc.start()

    # Measure execution time
    ray_ds = dataset.get_ray_dataset(size=size, start_idx=start_idx, **kwargs)#.select_columns(
        #["idx", "loc", "time_data"])
    data_gen = iter(ray_ds.iter_rows())
    first_element = next(data_gen)  # Force evaluation of the first element
    print(first_element)
    start_time = time.time()
    for i in data_gen:
        pass
    end_time = time.time()

    # Measure memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print results
    print(f"Execution time for {dataset_name}: {end_time - start_time:.2f} seconds")
    print(f"Current memory usage for {dataset_name}: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage for {dataset_name}: {peak / 1024 / 1024:.2f} MB")
    print(ray_ds.stats())

def measure_performance_generator(dataset, dataset_name, size, start_idx, **kwargs):
    """
    Measure execution time and memory usage for a generator-based dataset implementation.

    Args:
        dataset: The dataset instance to test.
        dataset_name: Name of the dataset implementation (for reporting).
        size: Number of samples to generate.
        start_idx: Starting index for the dataset.
        **kwargs: Additional arguments for dataset generation.

    Returns
    -------
        None
    """
    print(f"\nTesting {dataset_name} (Generator)...")

    # Start memory tracing
    tracemalloc.start()

    # Measure execution time
    data_gen = dataset.generate(size=size, start_idx=start_idx, **kwargs)
    first_element = next(data_gen)  # Force evaluation of the first element
    start_time = time.time()
    for _ in data_gen:
        pass
    end_time = time.time()

    # Measure memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print results
    print(f"Execution time for {dataset_name}: {end_time - start_time:.2f} seconds")
    print(f"Current memory usage for {dataset_name}: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage for {dataset_name}: {peak / 1024 / 1024:.2f} MB")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare DatasetSynthetic implementations.")
    parser.add_argument(
        "--implementation",
        choices=["original", "new"],
        required=True,
        help="Choose which implementation to test: 'original' (DatasetSynthetic) or 'new' (NewDatasetSynthetic).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=30,
        help="Number of samples to generate (default: 30).",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=4,
        help="Number of tasks for parallel processing (default: 4).",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for the dataset (default: 0).",
    )
    args = parser.parse_args()

    # Test the selected implementation
    if args.implementation == "original":
        # Test original DatasetSynthetic (Generator)
        ds = DatasetSynthetic(mic_sig_noise=False, mic_pos_noise=False, tasks=args.tasks)
        measure_performance_generator(
            ds,
            "DatasetSynthetic",
            args.size,
            args.start_idx,
            split="training",
            features=["idx", "loc", "time_data"],
        )
    elif args.implementation == "new":
        # Test NewDatasetSynthetic (Ray Dataset)
        newds = NewDatasetSynthetic(mic_pos_noise=False)
        newds.signal_model.dtype = "float64"
        measure_performance_ray(newds, "NewDatasetSynthetic", args.size, args.start_idx, tasks=args.tasks)


if __name__ == "__main__":
    main()




