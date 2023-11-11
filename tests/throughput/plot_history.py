import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set() # Setting seaborn as default style even if use only matplotlib

sns.set_theme(style="whitegrid")


def plot_all_features_over_tasks(name, device="compute4"):
    if device == "compute4":
        dev_name = "24x Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz"
    else:
        raise NotImplementedError
    file = Path(__file__).parent.absolute() / "results" / name
    df = pd.read_pickle(file)
    dataset_names = df.dataset.unique()
    frequencies = df.f.unique()
    modes = df["mode"].unique()
    ncols = modes.shape[0]
    for dataset_name in dataset_names:
        for f in frequencies:
            fig, axes = plt.subplots(1, ncols, figsize=(ncols*4, 4), sharey=False)
            if f is None:
                fig.suptitle(f"{dataset_name} (all frequencies) @ {dev_name}")
            else:
                fig.suptitle(f"{dataset_name} (single frequency, f={f} Hz) @ {dev_name}")
            for n, mode in enumerate(modes):
                key_names = ["dataset","method","f","mode"]
                keys = [dataset_name, "generate", f, mode]
                df_plot = df[df[key_names].isin(keys).all(1)]
                df_plot = df_plot[df_plot["feature"] != "None"]
                if len(df_plot) > 0:
                    sns.barplot(
                        ax = axes[n],
                        data = df_plot,
                        errorbar=None,
                        #kind="bar",
                        x="tasks", y="throughput", hue="feature")
                    axes[n].set_title(f"calculation mode: {mode}")
                    axes[n].set_xlabel("concurrent tasks")
                    axes[n].set_ylabel("throughput (source cases/s)")
                    name = os.path.splitext(name)[0]
            fig.tight_layout()
            fig.savefig(Path(__file__).parent.parent.parent.absolute() / "sphinx" / "source" / "_static" / f"{device}_all_features-over-tasks_{dataset_name}_f{f}.png")

def plot_feature_over_tasks(name):
    file = Path(__file__).parent.absolute() / "results" / name
    df = pd.read_pickle(file)
    dataset_names = df.dataset.unique()
    frequencies = df.f.unique()
    features = df.feature.unique()
    for dataset_name in dataset_names:
        for f in frequencies:
            for feature in features:
                key_names = ["dataset","method","f","feature"]
                keys = [dataset_name, "generate", f, feature]
                df_plot = df[df[key_names].isin(keys).all(1)]
                df_plot = df_plot[df_plot["feature"] != "None"]
                if len(df_plot) > 0:
                    pl = sns.catplot(
                        data = df_plot,
                        errorbar=None,
                        kind="bar",
                        x="tasks", y="throughput", hue="mode")
                    pl.fig.suptitle(f"{dataset_name} (feature={feature}, f={f} Hz)")
                    name = os.path.splitext(name)[0]
                    pl.savefig(Path(__file__).parent.absolute() / "results" / f"{name}_{feature}-over-tasks_{dataset_name}_f{f}.png")


if __name__ == "__main__":

    name = "throughput_c807562092a5_10-Nov-2023.pkl"
    plot_all_features_over_tasks(name)
    #plot_feature_over_tasks(name)
