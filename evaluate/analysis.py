import csv
import itertools
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ORDER = ["ae", "vae", "dcgan", "wgan_gp", "ddpm"]
NAMES = {"ae": "AE", "vae": "VAE", "dcgan": "DCGAN", "wgan_gp": "WGAN-GP", "ddpm": "DDPM"}
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKERS = dict(zip(ORDER, "osD^v"))
COMPONENTS = {"cpu_j": ("CPU", "#2a78d6"), "ram_j": ("RAM", "#eb6834"), "gpu_j": ("GPU", "#1baf7a")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DONE = {"completed", "target_reached", "budget_exhausted"}
PLAIN = FuncFormatter(lambda v, _: f"{v:,g}")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK, "font.size": 10,
    "lines.linewidth": 2, "legend.frameon": False,
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
    fig.savefig(path)
    plt.close(fig)


def bar_chart(labels, values, xlabel, title, path, fmt):
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(labels) + 1.2))
    y = range(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=COLORS["ae"])
    ax.set(yticks=list(y), yticklabels=labels, xscale="log", xlabel=xlabel, title=title)
    ax.xaxis.set_major_formatter(PLAIN)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.annotate(fmt(v), (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK2)
    ax.margins(x=0.15)
    save(fig, path)


def plot_fid_vs_energy(summary, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s in summary:
        m, x, y = s["model"], s["train_total_j_mean"] / 1e3, s["final_fid_mean"]
        ax.errorbar(x, y, xerr=num(s["train_total_j_std"]) / 1e3, yerr=num(s["final_fid_std"]), fmt="none",
                    elinewidth=1, ecolor=COLORS[m])
        ax.plot(x, y, MARKERS[m], color=COLORS[m], ms=9, mec=SURFACE, mew=2, label=NAMES[m])
        ax.annotate(NAMES[m], (x, y), xytext=(9, 6), textcoords="offset points", color=INK2)
    ax.set(xscale="log", xlabel="Training energy (kJ, log scale)", ylabel="Final FID (lower is better)",
           title="Quality vs training energy")
    ax.xaxis.set_major_formatter(PLAIN)
    ax.margins(x=0.15, y=0.1)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
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
    ax.set_title("Training energy by component", pad=28)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1)
    save(fig, path)


def mean_curve(curves):
    n = min(len(c) for c in curves)
    return [statistics.mean(c[i] for c in curves) for i in range(n)]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_lines(series, xlabel, ylabel, title, path, hline=None, log=True):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, (x, y) in series.items():
        ax.plot(x, y, color=COLORS[m], marker=MARKERS[m], ms=5, label=NAMES[m])
    if hline is not None:
        ax.axhline(hline, color=MUTED, linewidth=1)
        ax.annotate(f"target FID {hline:g}", (0, hline), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", color=MUTED)
    ax.set(yscale="log" if log else "linear", xlabel=xlabel, ylabel=ylabel, title=title)
    ax.yaxis.set_major_formatter(PLAIN)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
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
    rows = [run_row(r) for r in runs]
    summary = summarize(rows)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "summary.csv", summary)

    plot_fid_vs_energy(summary, plots / "fid_vs_energy.png")
    plot_energy_breakdown(summary, plots / "energy_breakdown.png")
    plot_lines(model_curves(runs, train_energy_curve), "Epoch", "Cumulative training energy (kJ, log scale)",
               "Training energy over epochs", plots / "cumulative_energy.png")
    infer = {}
    for r in runs:
        for sampler, v in r["inference"].items():
            infer.setdefault((r["model"], sampler), []).append(v["j_per_1k"])
    labels = [NAMES[m] if s == "direct" else f"{NAMES[m]} · {s}" for m, s in infer]
    bar_chart(labels, [statistics.mean(v) for v in infer.values()], "Energy per 1,000 images (J, log scale)",
              "Inference energy", plots / "inference_energy.png", lambda v: f"{v:,.1f} J")
    bar_chart([NAMES[s["model"]] for s in summary], [s["qpj_train_mean"] for s in summary],
              "QpJ = 1 / (FID × training energy in J), log scale", "Quality per Joule (higher is better)",
              plots / "qpj.png", lambda v: f"{v:.2e}")
    if all((r["dir"] / "fid_history.csv").exists() for r in runs):
        target = runs[0]["stopping"]["target_fid"]
        plot_lines(model_curves(runs, fid_curve), "Epoch", "FID at periodic check", "FID during training",
                   plots / "fid_vs_epoch.png", hline=target, log=False)
    print(f"Wrote {out / 'runs.csv'}, {out / 'summary.csv'} and plots in {plots}")
