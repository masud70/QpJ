"""RQ3 figure: cumulative inference energy relative to training energy, versus images generated.

Usage (from the repo root):
    python plot_breakeven.py results/target_quality_fid50_ddim100__native figures/rq3_breakeven_target.pdf
    python plot_breakeven.py results/fixed_budget__native figures/rq3_breakeven_fixed.pdf

Each line is N * e / E_train for one model (mean over seeds), where e is the inference energy
per image and E_train the training energy. A line crosses 1 at the break-even point N*.
"""
import glob
import json
import statistics
import sys

import numpy as np

sys.path.insert(0, "evaluate")
import analysis as A  # noqa: E402  (repo plotting style: colors, markers, rcParams)
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

group, out = sys.argv[1], sys.argv[2]
N = np.logspace(2, 9, 200)
fig, ax = plt.subplots(figsize=(6.4, 4.8))

for model in A.ORDER:
    runs = [json.load(open(p)) for p in glob.glob(f"{group}/runs/{model}/seed*/run.json")]
    runs = [r for r in runs if r["status"] in A.DONE]
    if not runs:
        continue
    e_train = statistics.mean(r["energy"]["train"]["total_j"] for r in runs)
    for sampler, style in (("direct", "-"), ("ancestral", "-"), ("ddim_100", "--")):
        if sampler not in runs[0]["inference"]:
            continue
        e_img = statistics.mean(r["inference"][sampler]["j_per_1k"] for r in runs) / 1000
        n_star = e_train / e_img
        label = A.NAMES[model] + {"ancestral": " (ancestral)", "ddim_100": " (DDIM-100)"}.get(sampler, "")
        ax.plot(N, N * e_img / e_train, style, color=A.COLORS[model], linewidth=2.8, label=label)
        ax.plot([n_star], [1], marker=A.MARKERS[model], color=A.COLORS[model], markersize=10, markeredgecolor=A.INK, markeredgewidth=1, zorder=5)

ax.axhspan(1, 1e4, color=A.AXIS, alpha=0.45, zorder=0, linewidth=0)
ax.axhline(1, color=A.INK, linestyle="-", linewidth=3.2, zorder=3)
ax.axvline(1e4, color=A.INK2, linestyle="--", linewidth=2.2, zorder=2)
ax.set(xscale="log", yscale="log", xlim=(1e2, 1e9), ylim=(1e-6, 1e4),
       xlabel="Images generated (log scale)", ylabel="Inference / training energy (log scale)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
models = ax.legend(fontsize=13, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.8)
ax.add_artist(models)
refs = [Line2D([], [], color=A.INK, linewidth=3.2, label="Break-even (inference = training)"),
        Patch(facecolor=A.AXIS, alpha=0.45, label="Inference > training"),
        Line2D([], [], color=A.INK2, linestyle="--", linewidth=2.2, label="10k evaluation images")]
ax.legend(handles=refs, fontsize=12, loc="lower right", frameon=True, facecolor=A.SURFACE, edgecolor=A.AXIS, framealpha=1)
fig.savefig(out, bbox_inches="tight")
print("saved", out)
