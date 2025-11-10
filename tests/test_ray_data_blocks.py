import acoular as ac # noqa: I001


from acoupipe.new_datasets.synthetic import DatasetSynthetic as NewDatasetSynthetic
print(ac.__version__)


# Test NewDatasetSynthetic (Ray Dataset)
newds = NewDatasetSynthetic(mic_pos_noise=False, signal_length=5.)
newds.signal_model.dtype = "float16"

# Measure execution time
ray_ds = newds.get_ray_dataset(features=["idx", "time_data"], size=5000, start_idx=0)#.select_columns(
#        ["idx"])
i = []
for row in ray_ds.iter_rows():
    i.append(row["idx"])
print(ray_ds.stats())
print(i)
print(row["meta"])
