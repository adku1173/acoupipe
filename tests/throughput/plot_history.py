import os

import pandas as pd
import seaborn as sns

fname = "throughput_29-Mar-2023.pkl"
node = "n07"

sns.set_theme(style="whitegrid")

data = pd.read_pickle(os.path.join("results",fname))
data = data[data["feature"] != "None"]
data = data[data["hostname"] == node]

d = data[data["dataset"] == "Dataset1"]
if len(d) > 0:
    pl = sns.catplot(
        data = d,
        errorbar=None,
        kind="bar",
        x="tasks", y="throughput", hue="feature")
    pl.fig.suptitle("Dataset1")
    pl.savefig(os.path.join("results",f"dataset1_{fname.split('.')[0]}.png"))


data = pd.read_pickle(os.path.join("results",fname))
data = data[data["feature"] != "None"]
data = data[data["hostname"] == node]
d = data[data["dataset"] == "Dataset2"]

if len(d) > 0:
    pl = sns.catplot(
        data = d,
        errorbar=None,
        kind="bar",
        x="tasks", y="throughput", hue="feature")
    pl.fig.suptitle("Dataset2")
    pl.savefig(os.path.join("results",f"dataset2_{fname.split('.')[0]}.png"))
