from acoupipe.datasets.synthetic import DatasetSynthetic1

feature = "csm"

dataset = DatasetSynthetic1(tasks=8)

gen = dataset.generate(features=[], f=100, num=3, split="training", size=10, progress_bar=False)
next(gen)

gen2 = dataset.generate(features=[], f=100, num=3, split="training", size=10, progress_bar=False)
data = next(gen2)

