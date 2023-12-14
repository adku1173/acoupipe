from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set() # Setting seaborn as default style even if use only matplotlib

sns.set_theme(style="whitegrid")

def plot_all_features_over_tasks(name, device):
    if device == "compute4":
        dev_name = "24x Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz"
    elif device == "compute1":
        dev_name = "16x Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz"
    else:
        dev_name = "{unknown device}"
    if isinstance(name, list):
        # combine dataframes
        df = pd.concat([pd.read_pickle(Path(__file__).parent.absolute() / "results" / n) for n in name])
    else:
        file = Path(__file__).parent.absolute() / "results" / name
        df = pd.read_pickle(file)
    dataset_names = df.dataset.unique()
    print(dataset_names)
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
                    if device == "compute1":
                        axes[n].set_xlabel("compute nodes")
                        axes[n].set_xticklabels(["2","4","8"])
            fig.tight_layout()
            fig.savefig(Path(__file__).parent.parent.parent.absolute() / "docs" / "source" / "_static" / f"{device}_all_features-over-tasks_{dataset_name}_f{f}.png")


if __name__ == "__main__":

    name = ["throughput_b0f7f4cef27e_28-Nov-2023.pkl"]
    plot_all_features_over_tasks(name, device="compute4")

    name = [
        "throughput_n00_12-Nov-2023.pkl",
        "throughput_n07_12-Nov-2023.pkl",
        "throughput_n11_12-Nov-2023.pkl",
    ]
    plot_all_features_over_tasks(name, device="compute1")
