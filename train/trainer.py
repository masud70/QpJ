import csv
import json
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

from config.config import ROOT
from data.celeba_loader import Batches, load_images
from evaluate.evaluate import generate, recon_mse, sample_for_benchmark, sync
from evaluate.fid import FID, reference_stats
from train import build_method
from utils.energy import make_meter
from utils.env_info import environment
from utils.flops import capacity
from utils.stopping import Progress, Stopper

DONE = {"completed", "target_reached", "budget_exhausted"}
FIELDS = ("duration_s", "cpu_j", "ram_j", "gpu_j", "total_j", "net_total_j")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CsvLog:
    def __init__(self, path):
        self.path, self.file, self.writer = path, None, None

    def write(self, row):
        if self.writer is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()


class Recorder:
    """Wraps every measured segment: writes energy_segments.csv and accumulates per-phase totals."""

    def __init__(self, meter, run_dir, progress, device):
        self.meter, self.progress, self.device = meter, progress, device
        self.csv = CsvLog(run_dir / "energy_segments.csv")
        self.totals, self.idle_w, self.last, self.count = {}, None, None, 0

    def measure_idle(self, seconds):
        self.meter.begin("idle")
        time.sleep(seconds)
        r = self.meter.end()
        self.idle_w = {k[:-2]: r[k] / r["duration_s"] for k in ("cpu_j", "ram_j", "gpu_j", "total_j")}

    @contextmanager
    def segment(self, phase, label):
        sync(self.device)
        self.meter.begin(label)
        yield
        sync(self.device)
        r = self.meter.end()
        r["net_total_j"] = r["total_j"] - self.idle_w["total"] * r["duration_s"]
        total = self.totals.setdefault(phase, {**dict.fromkeys(FIELDS, 0.0), "measured": dict(r["measured"])})
        for k in FIELDS:
            total[k] += r[k]
        total["measured"] = {c: total["measured"][c] and v for c, v in r["measured"].items()}
        p = self.progress
        self.csv.write({"idx": self.count, "phase": phase, "label": label, "step_end": p.steps,
                        "epoch_end": round(p.batches.epochs, 4), **{k: r[k] for k in FIELDS},
                        **{f"{c}_measured": v for c, v in r["measured"].items()}})
        self.last, self.count = r, self.count + 1


def train_loop(cfg, run_dir, method, batches, progress, stopper, rec, evaluate):
    log, history = CsvLog(run_dir / "progress.csv"), CsvLog(run_dir / "fid_history.csv")
    every, segment = cfg["experiment"]["log_every_steps"], 0
    while True:
        method.set_mode(True)
        start, t0 = progress.train_time_s, time.perf_counter()
        with rec.segment("train", f"train_{segment}"):
            while not stopper.segment_done(progress):
                losses = method.train_step(batches)
                progress.steps += 1
                progress.train_time_s = start + time.perf_counter() - t0
                if progress.steps % every == 0:
                    row = {**progress.as_dict(), "lr": method.lr(), **{k: round(float(v), 6) for k, v in losses.items()}}
                    log.write(row)
                    print(row, flush=True)
        progress.energy_j += rec.last["total_j"]
        fid = None
        if stopper.evaluates:
            method.set_mode(False)
            with rec.segment("eval", f"eval_{segment}"):
                fid = evaluate(cfg["stopping"]["target_quality"]["fid_samples"], method.eval_sampler)
            history.write({"eval_idx": segment, "step": progress.steps, "epoch": round(batches.epochs, 4),
                           "sampler": method.eval_sampler, "n_samples": cfg["stopping"]["target_quality"]["fid_samples"],
                           "fid": fid, "cum_train_total_j": rec.totals["train"]["total_j"],
                           "cum_eval_total_j": rec.totals["eval"]["total_j"]})
            print({"eval": segment, "fid": round(fid, 3)}, flush=True)
        segment += 1
        if stopper.end_segment(progress, fid):
            return


def execute(cfg, run_dir, summary):
    exp, ev, run, device = cfg["experiment"], cfg["evaluation"], cfg["run"], cfg["experiment"]["device"]
    torch.manual_seed(run["seed"])
    torch.backends.cudnn.benchmark = exp["cudnn_benchmark"]
    summary["environment"] = {**environment(cfg), "started_at": now()}
    images = load_images(cfg, "train")
    fid = FID(device, ev["batch_size"])
    ref = reference_stats(cfg, fid)
    method = build_method(run["model"], cfg, device)
    summary["capacity"] = cap = capacity(method, images, cfg["data"]["batch_size"])
    batches = Batches(images, cfg["data"]["batch_size"], device, run["seed"])
    progress, stopper = Progress(batches), Stopper(cfg)
    meter = make_meter(cfg, run_dir / "backend_raw")
    rec = Recorder(meter, run_dir, progress, device)
    rec.measure_idle(cfg["energy"]["idle_baseline_s"])

    def evaluate(n, sampler):
        method.prepare_sampling(images)
        return fid.score(generate(method, n, sampler, ev["batch_size"]), ref)

    train_loop(cfg, run_dir, method, batches, progress, stopper, rec, evaluate)
    method.set_mode(False)
    if run["model"] == "ae":
        with rec.segment("train", "gaussian_fit"):
            method.prepare_sampling(images)
    final_sampler = method.samplers[0]
    with rec.segment("final_sampling", f"final_{final_sampler}"):
        final = generate(method, ev["fid_samples"], final_sampler, ev["batch_size"])
    inference, inference_csv = {}, CsvLog(run_dir / "inference.csv")
    for s in method.samplers:
        with rec.segment("inference", f"inference_{s}"):
            n = sample_for_benchmark(method, s, cfg)
        r = rec.last
        inference[s] = {"j_per_1k": r["total_j"] / n * 1000, "s_per_1k": r["duration_s"] / n * 1000}
        inference_csv.write({"sampler": s, "n_images": n, "batch_size": ev["batch_size"],
                             **{k: r[k] for k in FIELDS[:5]}, **inference[s], "flops_per_image": cap["flops_per_image"][s]})
    meter.close()

    final_fid = fid.score(final, ref)
    save_image(final[:64].float() / 255, run_dir / "samples.png", nrow=8)
    torch.save(method.state(), run_dir / "model_final.pt")
    energy = {k: v for k, v in rec.totals.items() if k != "inference"}
    train_j, eval_j = energy["train"]["total_j"], energy.get("eval", {}).get("total_j", 0.0)
    final_j = energy["final_sampling"]["total_j"]
    summary.update({
        "status": stopper.status,
        "progress": progress.as_dict(),
        "idle_power_w": rec.idle_w,
        "energy": energy,
        "quality": {
            "final_fid": final_fid, "final_fid_samples": ev["fid_samples"], "final_sampler": final_sampler,
            "recon_mse": recon_mse(method, load_images(cfg, "test")[:ev["fid_samples"]], ev["batch_size"]),
            "target_reached_at": {"epoch": round(batches.epochs, 4), "train_total_j": train_j,
                                  "train_eval_total_j": train_j + eval_j} if stopper.status == "target_reached" else None,
        },
        "qpj": {"train": 1 / (final_fid * train_j), "train_eval": 1 / (final_fid * (train_j + eval_j)),
                "train_inference": 1 / (final_fid * (train_j + final_j))},
        "inference": inference,
    })
    summary["environment"]["ended_at"] = now()


def train_run(run_dir):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    mode = cfg["stopping"]["mode"]
    summary = {"model": cfg["run"]["model"], "seed": cfg["run"]["seed"], "backend": cfg["energy"]["backend"],
               "stopping": {"mode": mode, **cfg["stopping"][mode]}, "status": "failed"}
    try:
        execute(cfg, run_dir, summary)
    except BaseException:
        summary["error"] = traceback.format_exc()
        raise
    finally:
        (run_dir / "run.json").write_text(json.dumps(summary, indent=2))


def group_dir(cfg):
    return Path(cfg["experiment"]["results_dir"]) / f"{cfg['stopping']['mode']}__{cfg['energy']['backend']}"


def is_done(run_dir):
    path = run_dir / "run.json"
    return path.exists() and json.loads(path.read_text()).get("status") in DONE


def comparable(cfg):
    """Config without keys that select runs, only affect pacing, or belong to the inactive stopping mode."""
    skip, mode = {"models", "seeds", "cooldown_s"}, cfg["stopping"]["mode"]
    return {**{k: v for k, v in cfg.items() if k != "run"},
            "experiment": {k: v for k, v in cfg["experiment"].items() if k not in skip},
            "stopping": {"mode": mode, mode: cfg["stopping"][mode]}}


def run_suite(cfg, overwrite=False):
    exp, first = cfg["experiment"], True
    plan = [(seed, model, group_dir(cfg) / "runs" / model / f"seed{seed}")
            for seed in exp["seeds"] for model in exp["models"]]
    for _, _, run_dir in plan:
        if is_done(run_dir) and not overwrite:
            old = yaml.safe_load((run_dir / "config.yaml").read_text())
            if comparable(old) != comparable(cfg):
                raise SystemExit(f"{run_dir} finished with a different config. "
                                 "Use --overwrite or a different experiment.results_dir.")
    for seed, model, run_dir in plan:
        if is_done(run_dir) and not overwrite:
            print(f"[seed {seed}] {model}: already done, skipping")
            continue
        if not first:
            time.sleep(exp["cooldown_s"])
        first = False
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "run": {"model": model, "seed": seed}}))
        print(f"[seed {seed}] {model}: running -> {run_dir}", flush=True)
        start = time.time()
        with open(run_dir / "stdout.log", "w") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "run_all.py"), "_run", str(run_dir)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        result = run_dir / "run.json"
        status = json.loads(result.read_text())["status"] if result.exists() else "failed (killed)"
        print(f"[seed {seed}] {model}: {status} ({(time.time() - start) / 60:.1f} min)", flush=True)
