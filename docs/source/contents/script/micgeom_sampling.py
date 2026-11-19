"""Create the figures example_1.png ... example_4.png for the module example page.

The figures are produced by running ``examples/micgeom_sampling_example.py``, which is also
offered as a download on the example page. Each call to ``plt.show()`` in the example saves
the current figure instead of displaying it.
"""

import runpy
from pathlib import Path

# Configure OpenBLAS before matplotlib loads NumPy.
import acoular as ac  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[4]
dpath = Path(__file__).parent.parent.parent / "_static"
example = root / "examples" / "micgeom_sampling_example.py"

counter = {"n": 0}


def save_current_figure(*args, **kwargs):
    counter["n"] += 1
    fig = plt.gcf()
    fig.savefig(dpath / f"example_{counter['n']}.png", dpi=300)
    plt.close(fig)


plt.show = save_current_figure
runpy.run_path(str(example), run_name="__main__")
