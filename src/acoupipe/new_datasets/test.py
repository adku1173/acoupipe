



import time

from acoupipe.new_datasets.synthetic import DatasetSynthetic

# # Add RAY_DEBUG environment variable to enable Ray Debugger
# ray.init(runtime_env={
#     "env_vars": {"RAY_DEBUG": "1"},
# })

# Create a dataset with the default configuration

dataset = DatasetSynthetic(fs=44100, f=1000)
print(dataset.monte_carlo.mic_sig_noise)

map_fn = dataset._build_map_fn()

data = {"idx":0}
start_time = time.time()
data = map_fn(data)
end_time = time.time()

print(f"Function call took {end_time - start_time:.6f} seconds")
print(data.keys())


# # Generate a Ray dataset
# ray_ds = dataset.get_ray_dataset(size=5, features=["mic_csm", "loc"])
# for d in ray_ds.iter_rows():
#     print(d["mic_csm"].shape)

# tf_ds = ray_ds.to_tf(
#     feature_columns=["idx"],
#     label_columns=["idx"],
# )
# tf_ds = tf_ds.batch(1)
# for d in tf_ds:
#     print(d["source_csm"].shape)
#     print(d["idx"].shape)
