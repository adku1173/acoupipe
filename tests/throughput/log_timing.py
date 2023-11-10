from acoupipe.datasets.experimental import DatasetMIRACLE
from acoupipe.datasets.synthetic import DatasetSynthetic1

feature = "csm"

dataset = DatasetSynthetic1()
dataset = DatasetMIRACLE()
#dataset.logger.setLevel("INFO")

gen = dataset.generate(features=[feature,"f","num"], f=100, num=3, split="training", size=10, progress_bar=False)

data = next(gen)
print(data)

