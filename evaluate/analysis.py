import csv
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

ORDER = ["ae", "vae", "dcgan", "wgan_gp", "ddpm"]
NAMES = {"ae": "AE", "vae": "VAE", "dcgan": "DCGAN", "wgan_gp": "WGAN-GP", "ddpm": "DDPM"}
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(ORDER, "osD^v"))
COMPONENTS = {"cpu_j": ("CPU", "#2a78d6"), "ram_j": ("RAM", "#eb6834"), "gpu_j": ("GPU", "#1baf7a")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SHORT = {"ancestral": "A", "ddim_100": "D"}                 # DDPM sampler tags; full names go in the caption
DONE = {"completed", "target_reached", "budget_exhausted"}
PLAIN = FuncFormatter(lambda v, _: f"{v:,g}")
COMPACT = FuncFormatter(lambda v, _: next((f"{v / d:g}{u}" for u, d in (("M", 1e6), ("k", 1e3)) if v >= d), f"{v:g}"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK, "font.size": 14,
    "lines.linewidth": 3, "legend.frameon": False,
})


def load_runs(group):
    runs = []
    for path in group.glob("runs/*/seed*/run.json"):
        r = json.loads(path.read_text())
        if r["status"] in DONE:
            runs.append({**r, "dir": path.parent})
    return sorted(runs, key=lambda r: (ORDER.index(r["model"]), r["seed"]))


def run_row(r):
    e, q, cap, t = r["energy"], r["quality"], r["capacity"], r["energy"]["train"]
    final, target = q["final_sampler"], q["target_reached_at"]
    return {
        "model": r["model"], "seed": r["seed"], "status": r["status"], "params": cap["params"]["total"],
        "flops_train_step": cap["flops_per_train_step"], "flops_image": cap["flops_per_image"][final],
        **{k: r["progress"][k] for k in ("epochs", "steps", "train_time_s")},
        **{f"train_{k}": t[k] for k in ("cpu_j", "ram_j", "gpu_j", "total_j", "net_total_j")},
        "eval_total_j": e.get("eval", {}).get("total_j", 0.0), "final_fid": q["final_fid"],
        **{f"qpj_{k}": v for k, v in r["qpj"].items()},
        "energy_to_target_j": target["train_total_j"] if target else "",
        "inference_j_per_1k": r["inference"][final]["j_per_1k"],
    }


def summarize(rows):
    summary = []
    for model in ORDER:
        group = [r for r in rows if r["model"] == model]
        if not group:
            continue
        s = {"model": model, "n_runs": len(group), "n_target_reached": sum(r["status"] == "target_reached" for r in group)}
        for key in list(group[0])[3:]:
            values = [r[key] for r in group if r[key] != ""]
            s[f"{key}_mean"] = statistics.mean(values) if values else ""
            s[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(s)
    return summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def num(value):
    return value if value != "" else 0.0


def fmt_j(joules):
    for unit, scale in (("MJ", 1e6), ("kJ", 1e3)):
        if abs(joules) >= scale:
            return f"{joules / scale:,.1f} {unit}"
    return f"{joules:,.1f} J"


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plain_log(axis):
    """Plain-number tick labels on a log axis; on short ranges also label minor ticks at 2x and 5x."""
    lo, hi = sorted(axis.get_view_interval())
    axis.set_major_formatter(COMPACT if hi / lo >= 1e4 else PLAIN)
    some = FuncFormatter(lambda v, _: f"{v:,g}" if f"{v:e}"[0] in "25" else "")
    axis.set_minor_formatter(some if hi / lo < 100 else NullFormatter())


def legend_inside(ax, ncol=1):
    ax.legend(loc="best", ncol=ncol, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9,
              columnspacing=1, handlelength=1.5)


def bar_chart(labels, values, colors, xlabel, path, fmt):
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(labels) + 1.2))
    y = range(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=colors)
    ax.set(yticks=list(y), yticklabels=labels, xscale="log", xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.annotate(fmt(v), (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK2)
    ax.margins(x=0.15)
    plain_log(ax.xaxis)
    save(fig, path)


def plot_fid_vs_energy(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in summary:
        m, x, y = s["model"], s["train_total_j_mean"] / 1e3, s["final_fid_mean"]
        ax.errorbar(x, y, xerr=num(s["train_total_j_std"]) / 1e3, yerr=num(s["final_fid_std"]), fmt="none",
                    elinewidth=1, ecolor=COLORS[m])
        ax.plot(x, y, MARKERS[m], color=COLORS[m], ms=13, mec=SURFACE, mew=2, label=NAMES[m])
        ax.annotate(NAMES[m], (x, y), xytext=(9, 6), textcoords="offset points", color=INK2)
    ax.set(xscale="log", xlabel="Training energy (kJ, log scale)", ylabel="Final FID (lower is better)")
    ax.margins(x=0.15, y=0.1)
    plain_log(ax.xaxis)
    legend_inside(ax)
    save(fig, path)


def plot_energy_breakdown(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(summary) + 1.6))
    y = range(len(summary))[::-1]
    left = [0.0] * len(summary)
    for key, (label, color) in COMPONENTS.items():
        widths = [s[f"train_{key}_mean"] / 1e3 for s in summary]
        ax.barh(y, widths, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left = [a + b for a, b in zip(left, widths)]
    for yi, total in zip(y, left):
        ax.annotate(fmt_j(total * 1e3), (total, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK2)
    ax.set(yticks=list(y), yticklabels=[NAMES[s["model"]] for s in summary], xlabel="Training energy (kJ)")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    legend_inside(ax)
    save(fig, path)


def plot_energy_fid(summary, path):
    """Two bars per model, each metric normalized to its own maximum, real values printed on top."""
    metrics = [("train_total_j", "Training energy", None, fmt_j), ("final_fid", "Final FID", "//", lambda v: f"{v:.1f}")]
    models, width = [s["model"] for s in summary], 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for i, (key, label, hatch, fmt) in enumerate(metrics):
        means, stds = [s[f"{key}_mean"] for s in summary], [num(s[f"{key}_std"]) for s in summary]
        top = max(means)
        xs = [j + (i - 0.5) * width for j in range(len(models))]
        ax.bar(xs, [m / top * 100 for m in means], width, yerr=[sd / top * 100 for sd in stds], hatch=hatch,
               color=[COLORS[m] for m in models], edgecolor=SURFACE, linewidth=1,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        for x, m, sd in zip(xs, means, stds):
            ax.annotate(fmt(m), (x, (m + sd) / top * 100), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=13, color=INK2)
    handles = [Patch(facecolor=MUTED, edgecolor=SURFACE, hatch=h, label=l) for _, l, h, _ in metrics]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=16)
    ax.set(xticks=range(len(models)), xticklabels=[NAMES[m] for m in models], ylim=(0, 128))
    ax.set_ylabel("% of the highest model", fontsize=16)
    ax.set_yticks(range(0, 101, 25))
    ax.spines["left"].set_bounds(0, 100)
    ax.tick_params(labelsize=16)
    ax.grid(axis="x", visible=False)
    save(fig, path)


def mean_curve(curves):
    n = min(len(c) for c in curves)
    return [statistics.mean(c[i] for c in curves) for i in range(n)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_lines(series, xlabel, ylabel, path, hline=None, log=True):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, (x, y) in series.items():
        ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=7, label=NAMES[m])
    if hline is not None:
        ax.axhline(hline, color=MUTED, linewidth=1)
    ax.set(yscale="log" if log else "linear", xlabel=xlabel, ylabel=ylabel)
    if log:
        plain_log(ax.yaxis)
    legend_inside(ax, ncol=2)
    save(fig, path)


def train_energy_curve(run_dir):
    rows = [r for r in read_csv(run_dir / "energy_segments.csv") if r["phase"] == "train"]
    return [float(r["epoch_end"]) for r in rows], list(itertools.accumulate(float(r["total_j"]) / 1e3 for r in rows))


def fid_curve(run_dir):
    rows = read_csv(run_dir / "fid_history.csv")
    return [float(r["epoch"]) for r in rows], [float(r["fid"]) for r in rows]


def model_curves(runs, curve):
    series = {}
    for model in ORDER:
        curves = [curve(r["dir"]) for r in runs if r["model"] == model]
        if curves:
            series[model] = (curves[0][0][:min(len(c[0]) for c in curves)], mean_curve([c[1] for c in curves]))
    return series


def analyze(group):
    group = Path(group)
    runs = load_runs(group)
    if not runs:
        raise SystemExit(f"No finished runs in {group}")
    out = group / "summary"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for old in plots.glob("*.png"):
        old.unlink()
    rows = [run_row(r) for r in runs]
    summary = summarize(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "summary.csv", summary)

    plot_fid_vs_energy(summary, plots / "fid_vs_energy.pdf")
    plot_energy_breakdown(summary, plots / "energy_breakdown.pdf")
    plot_energy_fid(summary, plots / "energy_fid.pdf")
    plot_lines(model_curves(runs, train_energy_curve), "Epoch", "Cumulative energy (kJ, log scale)",
               plots / "cumulative_energy.pdf")
    infer = {}
    for r in runs:
        for sampler, v in r["inference"].items():
            infer.setdefault((r["model"], sampler), []).append(v["j_per_1k"])
    labels = [NAMES[m] if s == "direct" else f"{NAMES[m]}-{SHORT.get(s, s)}" for m, s in infer]
    bar_chart(labels, [statistics.mean(v) for v in infer.values()], [COLORS[m] for m, _ in infer],
              "Energy per 1,000 images (J, log scale)", plots / "inference_energy.pdf",
              lambda v: f"{v:,.1f} J")
    bar_chart([NAMES[s["model"]] for s in summary], [s["qpj_train_mean"] for s in summary],
              [COLORS[s["model"]] for s in summary], "QpJ, higher is better (log scale)",
              plots / "qpj.pdf", lambda v: f"{v:.2e}")
    if all((r["dir"] / "fid_history.csv").exists() for r in runs):
        target = runs[0]["stopping"]["target_fid"]
        plot_lines(model_curves(runs, fid_curve), "Epoch", "FID at periodic check",
                   plots / "fid_vs_epoch.pdf", hline=target, log=False)
    print(f"Wrote {out / 'runs.csv'}, {out / 'summary.csv'} and plots in {plots}")
