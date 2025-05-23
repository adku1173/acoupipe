from acoupipe.datasets.experimental import DatasetMIRACLE

feature = 'csm'

dataset = DatasetMIRACLE(tasks=1, mode='analytic')

gen = dataset.generate(features=[], f=100, num=3, split='training', size=1000, progress_bar=True)
# for _data in gen:
#     pass

# cProfile
import cProfile

gen = dataset.generate(features=[], f=100, num=3, split='training', size=1000, progress_bar=True)
next(gen)
with cProfile.Profile() as pr:
    next(gen)
    pr.print_stats('tottime')

# import tracemalloc
# tracemalloc.start()
# gen = dataset.generate(features=[], f=100, num=3, split="training", size=1000, progress_bar=True)
# next(gen)
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()


# gen2 = dataset.generate(features=[], f=100, num=3, split="training", size=10, progress_bar=False)
# data = next(gen2)
