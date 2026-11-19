"""Recompute DDPM's final sampling and final FID with DDIM-100 from saved checkpoints (no retraining).

Everything else in the group is copied unchanged, including both inference benchmarks.

Step 1, on the GPU machine: generate 10k DDIM-100 images per DDPM run, metered like the original final sampling.
    python recompute_ddim_final.py measure <group_dir> <measure_dir>
Step 2, anywhere: copy the group and swap in the measured DDPM values.
    python recompute_ddim_final.py build <group_dir> <measure_dir> <new_group_dir>
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLER = "ddim_100"
LABEL = f"final_{SAMPLER}"
FIELDS = ("duration_s", "cpu_j", "ram_j", "gpu_j", "total_j", "net_total_j")


def ddpm_runs(group):
    return sorted(p.parent for p in Path(group).glob("runs/ddpm/seed*/run.json"))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(group, out):
    import torch
    import yaml
    from torchvision.utils import save_image

    from evaluate.evaluate import generate, sync
    from evaluate.fid import FID, reference_stats
    from train import build_method
    from utils.display import log, setup_logging
    from utils.energy import make_meter
    from utils.env_info import require_gpu

    setup_logging()
    runs = ddpm_runs(group)
    if not runs:
        raise SystemExit(f"No DDPM runs found in {group} (expected runs/ddpm/seed*/run.json)")
    log.info(f"Measuring {len(runs)} DDPM runs from {group}: {', '.join(r.name for r in runs)}")
    for run in runs:
        dest = Path(out) / "ddpm" / run.name
        if (dest / "measure.json").exists():
            log.info(f"ddpm/{run.name}: already measured, skipping")
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        require_gpu(cfg)
        device, ev = cfg["experiment"]["device"], cfg["evaluation"]
        idle_w = json.loads((run / "run.json").read_text())["idle_power_w"]["total"]

        method = build_method("ddpm", cfg, device)
        state = torch.load(run / "model_final.pt", map_location=device)
        method.net.load_state_dict(state["unet"])
        method.ema.load_state_dict(state["ema"])
        method.set_mode(False)

        dest.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg["run"]["seed"])
        meter = make_meter(cfg, dest / "backend_raw")
        try:
            sync(device)
            meter.begin(LABEL)
            images = generate(method, ev["fid_samples"], SAMPLER, ev["batch_size"],
                              desc=f"ddpm/{run.name} final samples [{SAMPLER}]")
            sync(device)
            segment = meter.end()
        finally:
            meter.close()
        segment["net_total_j"] = segment["total_j"] - idle_w * segment["duration_s"]

        fid = FID(device, ev["batch_size"])
        score = fid.score(images, reference_stats(cfg, fid))
        save_image(images[:64].float() / 255, dest / "samples.png", nrow=8)
        result = {"sampler": SAMPLER, "n_samples": ev["fid_samples"], "final_fid": score, "segment": segment}
        (dest / "measure.json").write_text(json.dumps(result, indent=2))
        log.info(f"ddpm/{run.name}: FID {score:.2f} · {segment['total_j'] / 1e3:.1f} kJ in {segment['duration_s']:.0f} s")


def replace_segment(run, measured_dir, segment):
    rows = read_csv(run / "energy_segments.csv")
    row = next(r for r in rows if r["phase"] == "final_sampling")
    old_label, row["label"] = row["label"], LABEL
    row.update({k: segment[k] for k in FIELDS})
    row.update({f"{c}_measured": v for c, v in segment["measured"].items()})
    write_csv(run / "energy_segments.csv", rows)

    path = run / "backend_raw" / "segments.json"
    new_raw = json.loads((measured_dir / "backend_raw" / "segments.json").read_text())[0]
    segments = [new_raw if s["label"] == old_label else s for s in json.loads(path.read_text())]
    path.write_text(json.dumps(segments, indent=1))
    shutil.copytree(measured_dir / "backend_raw", run / "backend_raw" / LABEL)


def update_run(run, result):
    path = run / "run.json"
    r = json.loads(path.read_text())
    seg, q, e = result["segment"], r["quality"], r["energy"]
    e["final_sampling"] = {**{k: seg[k] for k in FIELDS}, "measured": seg["measured"]}
    q.update(final_fid=result["final_fid"], final_sampler=SAMPLER, final_fid_samples=result["n_samples"])
    fid, train_j, eval_j = q["final_fid"], e["train"]["total_j"], e.get("eval", {}).get("total_j", 0.0)
    r["qpj"] = {"train": 1 / (fid * train_j), "train_eval": 1 / (fid * (train_j + eval_j)),
                "train_inference": 1 / (fid * (train_j + e["final_sampling"]["total_j"]))}
    path.write_text(json.dumps(r, indent=2))


def build(group, out, new):
    group, out, new = Path(group), Path(out), Path(new)
    if new.exists():
        raise SystemExit(f"{new} already exists")
    results = {p.parent.name: json.loads(p.read_text()) for p in out.glob("ddpm/seed*/measure.json")}
    missing = {run.name for run in ddpm_runs(group)} - set(results)
    if missing:
        raise SystemExit(f"Not measured yet: {sorted(missing)}")
    shutil.copytree(group, new, ignore=shutil.ignore_patterns("*.pt", "summary"))
    for seed, result in sorted(results.items()):
        run, measured_dir = new / "runs" / "ddpm" / seed, out / "ddpm" / seed
        replace_segment(run, measured_dir, result["segment"])
        update_run(run, result)
        shutil.copy(measured_dir / "samples.png", run / "samples.png")
        print(f"ddpm/{seed}: FID {result['final_fid']:.2f}, final sampling {result['segment']['total_j'] / 1e3:.1f} kJ")
    print(f"Built {new}. Next: analyze it and rerun plot_breakeven.py.")


if __name__ == "__main__":
    commands = {"measure": (measure, 2), "build": (build, 3)}
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) != commands[sys.argv[1]][1] + 2:
        raise SystemExit(__doc__)
    commands[sys.argv[1]][0](*sys.argv[2:])
