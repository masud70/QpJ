import csv
import json
import math
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
from utils.display import (NAMES, bar, config_text, countdown, energy_text, joules, log, power_text, row,
                           setup_logging, si, stopping_text)
from utils.energy import make_meter
from utils.env_info import environment, gpu_summary
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
        countdown(seconds, "Idle baseline")
        r = self.meter.end()
        self.idle_w = {k[:-2]: r[k] / r["duration_s"] for k in ("cpu_j", "ram_j", "gpu_j", "total_j")}
        row("Idle power", power_text(self.idle_w, r["measured"]))

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


def segment_label(stopper, k):
    seg, budget = stopper.segment, stopper.budget
    total = f"/{math.ceil(budget['value'] / seg['value'])}" if seg["unit"] == budget["unit"] else ""
    return f"{'Epoch' if seg['unit'] == 'epochs' else 'Segment'} {k}{total}"


def segment_steps(stopper, n_images, images_per_step):
    unit, value = stopper.segment["unit"], stopper.segment["value"]
    return {"epochs": math.ceil(value * n_images / images_per_step), "steps": value,
            "images": math.ceil(value / images_per_step)}.get(unit)


def train_loop(cfg, run_dir, method, batches, progress, stopper, rec, evaluate):
    log_csv, history = CsvLog(run_dir / "progress.csv"), CsvLog(run_dir / "fid_history.csv")
    every, segment, losses_text = cfg["experiment"]["log_every_steps"], 0, ""
    total = segment_steps(stopper, len(batches.images), batches.batch_size * method.batches_per_step)
    while True:
        method.set_mode(True)
        label = segment_label(stopper, segment + 1)
        start, t0 = progress.train_time_s, time.perf_counter()
        with rec.segment("train", f"train_{segment}"), bar(desc=label, total=total, unit="step", leave=False) as pbar:
            while not stopper.segment_done(progress):
                losses = method.train_step(batches)
                progress.steps += 1
                progress.train_time_s = start + time.perf_counter() - t0
                pbar.update()
                if progress.steps % every == 0:
                    values = {k: float(v) for k, v in losses.items()}
                    log_csv.write({**progress.as_dict(), "lr": method.lr(), **{k: round(v, 6) for k, v in values.items()}})
                    pbar.set_postfix({k: f"{v:.4g}" for k, v in values.items()})
                    losses_text = " · " + " · ".join(f"{k} {v:.4g}" for k, v in values.items())
        progress.energy_j += rec.last["total_j"]
        log.info(f"{label} done · {rec.last['duration_s']:.0f} s · {energy_text(rec.last)}{losses_text}")
        fid = None
        if stopper.evaluates:
            n = cfg["stopping"]["target_quality"]["fid_samples"]
            method.set_mode(False)
            with rec.segment("eval", f"eval_{segment}"):
                fid = evaluate(n, method.eval_sampler)
            history.write({"eval_idx": segment, "step": progress.steps, "epoch": round(batches.epochs, 4),
                           "sampler": method.eval_sampler, "n_samples": n, "fid": fid,
                           "cum_train_total_j": rec.totals["train"]["total_j"],
                           "cum_eval_total_j": rec.totals["eval"]["total_j"]})
        segment += 1
        stop = stopper.end_segment(progress, fid)
        if fid is not None:
            outcome = stopper.status.replace("_", " ") if stop else "continue"
            log.info(f"FID check {segment}: {fid:.2f} (target {stopper.target:g}, {method.eval_sampler}) → {outcome}")
        if stop:
            return


def run_header(cfg, method, cap, images, run_dir):
    model, device, data = cfg["run"]["model"], cfg["experiment"]["device"], cfg["data"]
    networks = ", ".join(f"{name} {si(p)}" for name, p in cap["params"]["per_network"].items())
    steps = len(images) // (data["batch_size"] * method.batches_per_step)
    source = "CelebA" if data["source"] == "celeba" else "synthetic"
    row("Device", f"GPU ✓ {torch.cuda.get_device_name(device)} ({device})" if device.startswith("cuda")
        else "CPU only (no GPU)")
    row("Model", f"{NAMES[model]} · {si(cap['params']['total'])} params ({networks}) · "
                 f"{si(cap['flops_per_train_step'], 'FLOPs')} per training step")
    row("Config", config_text(cfg[model]))
    row("Data", f"{source} {data['image_size']}×{data['image_size']} · {len(images):,} train images · "
                f"batch {data['batch_size']} · {steps:,} steps per epoch")
    row("Stopping", stopping_text(cfg))
    row("Energy", f"{cfg['energy']['backend']} backend")
    row("Output", run_dir)


def execute(cfg, run_dir, summary):
    exp, ev, run, device = cfg["experiment"], cfg["evaluation"], cfg["run"], cfg["experiment"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("experiment.device is 'cuda' but PyTorch sees no GPU. Check the NVIDIA driver and the "
                         "PyTorch CUDA build, or run with --set experiment.device=cpu.")
    torch.manual_seed(run["seed"])
    torch.backends.cudnn.benchmark = exp["cudnn_benchmark"]
    summary["environment"] = {**environment(cfg), "started_at": now()}
    log.info("Loading data and FID reference statistics")
    images = load_images(cfg, "train")
    fid = FID(device, ev["batch_size"])
    ref = reference_stats(cfg, fid)
    method = build_method(run["model"], cfg, device)
    summary["capacity"] = cap = capacity(method, images, cfg["data"]["batch_size"])
    run_header(cfg, method, cap, images, run_dir)
    batches = Batches(images, cfg["data"]["batch_size"], device, run["seed"])
    progress, stopper = Progress(batches), Stopper(cfg)
    meter = make_meter(cfg, run_dir / "backend_raw")
    rec = Recorder(meter, run_dir, progress, device)
    rec.measure_idle(cfg["energy"]["idle_baseline_s"])

    def evaluate(n, sampler):
        method.prepare_sampling(images)
        return fid.score(generate(method, n, sampler, ev["batch_size"], desc=f"FID check [{sampler}]"), ref)

    train_loop(cfg, run_dir, method, batches, progress, stopper, rec, evaluate)
    method.set_mode(False)
    if run["model"] == "ae":
        log.info("Fitting the latent Gaussian used for AE sampling")
        with rec.segment("train", "gaussian_fit"):
            method.prepare_sampling(images)
    final_sampler = method.samplers[0]
    with rec.segment("final_sampling", f"final_{final_sampler}"):
        final = generate(method, ev["fid_samples"], final_sampler, ev["batch_size"],
                         desc=f"Final samples [{final_sampler}]")
    log.info(f"Final samples: {ev['fid_samples']:,} images [{final_sampler}] · {energy_text(rec.last)}")
    inference, inference_csv = {}, CsvLog(run_dir / "inference.csv")
    for s in method.samplers:
        with rec.segment("inference", f"inference_{s}"):
            n = sample_for_benchmark(method, s, cfg)
        r = rec.last
        inference[s] = {"j_per_1k": r["total_j"] / n * 1000, "s_per_1k": r["duration_s"] / n * 1000}
        inference_csv.write({"sampler": s, "n_images": n, "batch_size": ev["batch_size"],
                             **{k: r[k] for k in FIELDS[:5]}, **inference[s], "flops_per_image": cap["flops_per_image"][s]})
        log.info(f"Inference [{s}]: {joules(inference[s]['j_per_1k'])} and {inference[s]['s_per_1k']:.1f} s "
                 f"per 1,000 images ({n:,} images measured)")
    meter.close()

    log.info("Computing final FID and saving outputs")
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
    log.info(f"Done · {stopper.status.replace('_', ' ')} · {batches.epochs:.1f} epochs · "
             f"{progress.train_time_s / 60:.1f} min training · {joules(train_j)} · FID {final_fid:.2f} · "
             f"QpJ {summary['qpj']['train']:.2e}")


def train_run(run_dir):
    run_dir = Path(run_dir)
    setup_logging(run_dir / "stdout.log")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    mode = cfg["stopping"]["mode"]
    summary = {"model": cfg["run"]["model"], "seed": cfg["run"]["seed"], "backend": cfg["energy"]["backend"],
               "stopping": {"mode": mode, **cfg["stopping"][mode]}, "status": "failed"}
    try:
        execute(cfg, run_dir, summary)
    except KeyboardInterrupt:
        summary["error"] = "interrupted"
        log.info("Interrupted")
        raise
    except BaseException as e:
        summary["error"] = str(e) if isinstance(e, SystemExit) else traceback.format_exc()
        log.error(f"Run failed:\n{summary['error']}")
        raise SystemExit(1)
    finally:
        (run_dir / "run.json").write_text(json.dumps(summary, indent=2))


def group_dir(cfg):
    return Path(cfg["experiment"]["results_dir"]) / f"{cfg['stopping']['mode']}__{cfg['energy']['backend']}"


def read_run(run_dir):
    path = run_dir / "run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def is_done(run_dir):
    return read_run(run_dir).get("status") in DONE


def comparable(cfg):
    """Config without keys that select runs, only affect pacing, or belong to the inactive stopping mode."""
    skip, mode = {"models", "seeds", "cooldown_s"}, cfg["stopping"]["mode"]
    return {**{k: v for k, v in cfg.items() if k != "run"},
            "experiment": {k: v for k, v in cfg["experiment"].items() if k not in skip},
            "stopping": {"mode": mode, mode: cfg["stopping"][mode]}}


def suite_header(cfg, plan, done):
    exp, device = cfg["experiment"], cfg["experiment"]["device"]
    gpu = gpu_summary(cfg["energy"]["gpu_index"]) if device.startswith("cuda") else None
    if device.startswith("cuda") and gpu is None:
        raise SystemExit("experiment.device is 'cuda' but nvidia-smi found no GPU. "
                         "Check the NVIDIA driver, or run with --set experiment.device=cpu.")
    log.info("Quality per Joule v2 · train")
    row("Backend", cfg["energy"]["backend"])
    row("Stopping", stopping_text(cfg))
    row("Device", f"GPU ✓ {gpu}" if gpu else "CPU only (experiment.device=cpu)")
    row("Runs", f"{len(exp['models'])} models × {len(exp['seeds'])} seeds = {len(plan)} "
                f"({done} already done, {len(plan) - done} to run)")
    row("Results", group_dir(cfg))


def suite_table(plan):
    log.info(f"\n{'Model':<9}{'Seed':>6}  {'Status':<18}{'Train time':>11}{'Train energy':>14}{'FID':>9}")
    for seed, model, run_dir in plan:
        r = read_run(run_dir)
        minutes = f"{r['progress']['train_time_s'] / 60:.1f} min" if "progress" in r else "-"
        energy = joules(r["energy"]["train"]["total_j"]) if "energy" in r else "-"
        fid = f"{r['quality']['final_fid']:.2f}" if "quality" in r else "-"
        log.info(f"{NAMES[model]:<9}{seed:>6}  {r.get('status', 'not run'):<18}{minutes:>11}{energy:>14}{fid:>9}")


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
    todo = [p for p in plan if overwrite or not is_done(p[2])]
    suite_header(cfg, plan, len(plan) - len(todo))
    for i, (seed, model, run_dir) in enumerate(todo, 1):
        if not first:
            countdown(exp["cooldown_s"], "Cooldown")
        first = False
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "run": {"model": model, "seed": seed}}))
        log.info(f"\n━━━ Run {i}/{len(todo)} · {NAMES[model]} · seed {seed} ━━━")
        subprocess.run([sys.executable, "-u", str(ROOT / "run_all.py"), "_run", str(run_dir)], cwd=ROOT)
        if not read_run(run_dir):
            log.info(f"Run was killed before writing results (see {run_dir / 'stdout.log'})")
    suite_table(plan)
