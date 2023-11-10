import os
from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_all_features_over_tasks(name):
    file = Path(__file__).parent.absolute() / "results" / name
    df = pd.read_pickle(file)
    dataset_names = df.dataset.unique()
    frequencies = df.f.unique()
    modes = df["mode"].unique()
    for dataset_name in dataset_names:
        for f in frequencies:
            for mode in modes:
                key_names = ["dataset","method","f","mode"]
                keys = [dataset_name, "generate", f, mode]
                df_plot = df[df[key_names].isin(keys).all(1)]
                df_plot = df_plot[df_plot["feature"] != "None"]
                if len(df_plot) > 0:
                    pl = sns.catplot(
                        data = df_plot,
                        errorbar=None,
                        kind="bar",
                        x="tasks", y="throughput", hue="feature")
                    pl.fig.suptitle(f"{dataset_name} (mode={mode}, f={f} Hz)")
                    name = os.path.splitext(name)[0]
                    pl.savefig(Path(__file__).parent.absolute() / "results" / f"{name}_all_features-over-tasks_{dataset_name}_{mode}_f{f}.png")

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

    name = "throughput_compute4_09-Nov-2023.pkl"
    plot_all_features_over_tasks(name)
    plot_feature_over_tasks(name)
