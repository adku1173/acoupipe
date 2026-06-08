from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt

from acoupipe.datasets.experimental import DatasetMIRACLE, DatasetSRIRACHA
from acoupipe.datasets.synthetic import DatasetSynthetic

f = 2000
mode = "wishart"
max_nsources = 10
sriracha_path = None
mic_sig_noise = False

fig, axs = plt.subplots(2, 4, figsize=(12, 9), sharey=True, sharex=True)
axs = axs.ravel()
fig.suptitle(f"Sourcemap ($f={f}$ Hz)", fontsize=12)

for i, scenario in enumerate(["SR2","Synthetic", "A1", "SRA1", "SR1", "A2", "R2", "SRA2"]):
#for i, scenario in enumerate(["A1"]):
    if scenario == "Synthetic":
        dataset = DatasetSynthetic(mode=mode, random_signal_length=False,  max_nsources=max_nsources, mic_sig_noise=mic_sig_noise)
    elif scenario in ["A1", "A2", "R2"]:
        dataset = DatasetMIRACLE(scenario=scenario, mode=mode,  max_nsources=max_nsources, mic_sig_noise=mic_sig_noise)
    else:
        dataset = DatasetSRIRACHA(scenario=scenario, mode=mode, srir_dir=sriracha_path, max_nsources=max_nsources, mic_sig_noise=mic_sig_noise)
    print(f"Processing dataset: {dataset.__class__.__name__}, scenario: {scenario}")
    data_generator = dataset.generate(
        features=["sourcemap", "loc", "f"],
        split="training",
        size=1,
        f=[f],
        num=0,
        start_idx=2,
    )
    data_sample = next(data_generator)

    extent = dataset.config.grid.extent

    # sound pressure level
    Lm = ac.L_p(data_sample["sourcemap"]).T
    Lm_max = Lm.max()
    Lm_min = Lm.max() - 20

    if scenario == "Synthetic":
        title = f"{dataset.__class__.__name__}"
    else:
        title = f"{dataset.__class__.__name__}: {scenario}"

    ax = axs[i]
    ax.set_title(title)
    im = ax.imshow(
        Lm,
        vmax=Lm_max,
        vmin=Lm_min,
        origin="lower",
        interpolation="bicubic",
        cmap="hot_r",
        extent=extent,
    )

    # plot source locations
    for loc in data_sample["loc"].T:
        ax.scatter(loc[0], loc[1], s=2, marker="x", color="cyan")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.tick_params(axis="both", which="both", direction="in")

    # colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, location="bottom", label="Sound Pressure Level (dB)")

fig.tight_layout()

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "all_datasets.png", dpi=300)
