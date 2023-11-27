from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt

from acoupipe.datasets.experimental import DatasetMIRACLE

srir_dir = "/home/kujawski/compute4/IR_AKAP/published"
f = 4000

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)
fig.suptitle(f"Sourcemap ($f={f}$ Hz)", fontsize=12)

for i, scenario in enumerate(["A1", "A2", "R2"]):

    dataset = DatasetMIRACLE(srir_dir=srir_dir, scenario=scenario, mode="wishart")
    data_generator = dataset.generate(features=["sourcemap","loc", "f"],
                                        split="training", size=1, f=[f], num=0, start_idx=1)
    data_sample = next(data_generator)

    extent = dataset.config.grid.extend()

    # sound pressure level
    Lm = ac.L_p(data_sample["sourcemap"]).T
    if i == 0:
        Lm_max = Lm.max()
        Lm_min = Lm.max() - 20

    axs[i].set_title(f"scenario: {scenario}")
    im = axs[i].imshow(Lm, vmax=Lm_max, vmin=Lm_min, extent=extent, origin="lower",interpolation="bicubic")

    # plot source locations
    for loc in data_sample["loc"].T:
        axs[i].scatter(loc[0], loc[1], s=1)
    axs[i].set_xlabel("x (m)")
    axs[i].set_ylabel("y (m)")

fig.colorbar(im, label="Sound Pressure Level (dB)", ax=axs[i])
fig.tight_layout()

dpath = Path(__file__).parent.parent.parent / "_static"
fig.savefig(dpath / "exp_sourcemap_example.png", dpi=300)
