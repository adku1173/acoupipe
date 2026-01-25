from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt

from acoupipe.datasets.experimental import DatasetMIRACLE, DatasetSRIRACHA
from acoupipe.datasets.synthetic import DatasetSynthetic

f = 2000
sriracha_path = "/home/kujawski/Documents/Projekte/DFG/AP5/miracle2/data/published"

fig, axs = plt.subplots(2, 4, figsize=(12, 9), sharey=True, sharex=True)
axs = axs.ravel()
fig.suptitle(f"Sourcemap ($f={f}$ Hz, 1/3 octave)", fontsize=12)

for i, scenario in enumerate(["Synthetic", "A1", "SRA1", "SR1", "A2", "R2", "SRA2", "SR2"]):
#for i, scenario in enumerate(["A1"]):
    if scenario == "Synthetic":
        dataset = DatasetSynthetic(mode="wishart")
    elif scenario in ["A1", "A2", "R2"]:
        dataset = DatasetMIRACLE(scenario=scenario, mode="wishart", srir_dir=sriracha_path)
    else:
        dataset = DatasetSRIRACHA(scenario=scenario, mode="wishart", srir_dir=sriracha_path)
    data_generator = dataset.generate(
        features=["sourcemap", "loc", "f"],
        split="training",
        size=1,
        f=[f],
        num=1,
        start_idx=2,
    )
    data_sample = next(data_generator)

    extent = dataset.config.grid.extent

    # sound pressure level
    Lm = ac.L_p(data_sample["sourcemap"]).T
    Lm_max = Lm.max()
    Lm_min = Lm.max() - 10

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
