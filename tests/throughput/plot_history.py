import os

import pandas as pd
import seaborn as sns

fname = "throughput_04-Feb-2023.pkl"

data = pd.read_pickle(os.path.join("results",fname))
sns.set_theme(style="whitegrid")

pl = sns.catplot(
    data = data[data["dataset"] == "Dataset1"],
    errorbar=None,
    kind="bar",
    x="tasks", y="throughput", hue="feature")
pl.fig.suptitle("Dataset1")
pl.savefig(os.path.join("results",f"dataset1_{fname.split('.')[0]}.png"))

pl = sns.catplot(
    data = data[data["dataset"] == "Dataset2"],
    errorbar=None,
    kind="bar",
    x="tasks", y="throughput", hue="feature")
pl.fig.suptitle("Dataset2")
pl.savefig(os.path.join("results",f"dataset2_{fname.split('.')[0]}.png"))
