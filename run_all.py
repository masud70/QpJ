import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from config.config import CHOICES, MODELS, load_config

BACKENDS, STOPPING = sorted(CHOICES[("energy", "backend")]), sorted(CHOICES[("stopping", "mode")])


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML file merged over config/default.yaml")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    common.add_argument("--backend", choices=BACKENDS, help="energy backend (default: energy.backend)")
    parser = argparse.ArgumentParser(description="Quality per Joule v2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", parents=[common], help="build data cache and FID reference statistics")
    sub.add_parser("check", parents=[common], help="test the energy backend under idle and load")
    train = sub.add_parser("train", parents=[common], help="train, evaluate and measure all runs")
    analyze = sub.add_parser("analyze", parents=[common], help="aggregate runs into summary tables and plots")
    for p in (train, analyze):
        p.add_argument("--stopping", choices=STOPPING, help="stopping criterion (default: stopping.mode)")
        p.add_argument("--target-fid", type=float, help="target_quality only (default: stopping.target_quality.target_fid)")
    train.add_argument("--models", nargs="+", choices=sorted(MODELS), help="default: experiment.models")
    train.add_argument("--seeds", nargs="+", type=int, help="default: experiment.seeds")
    train.add_argument("--overwrite", action="store_true", help="re-run finished runs")
    return parser.parse_args()


def config_from(args):
    overrides = list(args.set)
    if args.backend:
        overrides.append(f"energy.backend={args.backend}")
    if getattr(args, "stopping", None):
        overrides.append(f"stopping.mode={args.stopping}")
    if getattr(args, "target_fid", None) is not None:
        overrides.append(f"stopping.target_quality.target_fid={args.target_fid}")
    if getattr(args, "models", None):
        overrides.append(f"experiment.models=[{', '.join(args.models)}]")
    if getattr(args, "seeds", None):
        overrides.append(f"experiment.seeds=[{', '.join(map(str, args.seeds))}]")
    return load_config(args.config, overrides)


def require_gpu(cfg):
    if cfg["experiment"]["device"].startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("experiment.device is 'cuda' but PyTorch sees no GPU. Check the NVIDIA driver and the "
                         "PyTorch CUDA build, or run with --set experiment.device=cpu.")


def prepare(cfg):
    from data.celeba_loader import load_images
    from evaluate.fid import FID, reference_stats
    from utils.display import log, row
    require_gpu(cfg)
    log.info("Quality per Joule v2 · prepare")
    row("Data", f"{cfg['data']['source']} → {cfg['data']['root']}")
    row("Cache", cfg["data"]["cache_dir"])
    for split in ("train", "test"):
        row(split, f"{len(load_images(cfg, split)):,} images ready")
    reference_stats(cfg, FID(cfg["experiment"]["device"], cfg["evaluation"]["batch_size"]))
    log.info("Preparation complete")


def check(cfg, seconds=10):
    from evaluate.evaluate import sync
    from utils.display import bar, log, row
    from utils.energy import make_meter
    from utils.env_info import environment
    require_gpu(cfg)
    device = cfg["experiment"]["device"]
    log.info(f"Quality per Joule v2 · check · {cfg['energy']['backend']} backend")
    for key, value in environment(cfg).items():
        row(key, value)
    meter = make_meter(cfg, Path(cfg["experiment"]["results_dir"]) / f"check__{cfg['energy']['backend']}")
    x = torch.randn(2048, 2048, device=device)

    def work(label):
        for _ in bar(range(seconds), desc=f"Measuring {label}", leave=False,
                     bar_format="{desc}: {bar} {n_fmt}/{total_fmt} s"):
            start = time.time()
            while time.time() - start < 1:
                if label == "idle":
                    time.sleep(0.1)
                else:
                    (x @ x).tanh_()
            sync(device)

    for label in ("idle", "load"):
        meter.begin(label)
        work(label)
        r = meter.end()
        watts = {k[:-2]: round(r[k] / r["duration_s"], 1) for k in ("cpu_j", "ram_j", "gpu_j", "total_j")}
        log.info(f"{label:5s} W={watts} measured={r['measured']}")
    meter.close()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_run":
        from train.trainer import train_run
        return train_run(sys.argv[2])
    from utils.display import setup_logging
    setup_logging()
    args = parse_args()
    cfg = config_from(args)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "check":
        check(cfg)
    elif args.command == "train":
        from train.trainer import run_suite
        run_suite(cfg, args.overwrite)
    else:
        from evaluate.analysis import analyze
        from train.trainer import group_dir
        analyze(group_dir(cfg))


if __name__ == "__main__":
    main()
