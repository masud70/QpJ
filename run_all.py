import argparse
import json
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
    if getattr(args, "models", None):
        overrides.append(f"experiment.models=[{', '.join(args.models)}]")
    if getattr(args, "seeds", None):
        overrides.append(f"experiment.seeds=[{', '.join(map(str, args.seeds))}]")
    return load_config(args.config, overrides)


def prepare(cfg):
    from data.celeba_loader import load_images
    from evaluate.fid import FID, reference_stats
    for split in ("train", "test"):
        print(f"{split}: {tuple(load_images(cfg, split).shape)}")
    reference_stats(cfg, FID(cfg["experiment"]["device"], cfg["evaluation"]["batch_size"]))
    print(f"FID reference statistics cached in {cfg['data']['cache_dir']}")


def check(cfg):
    from evaluate.evaluate import sync
    from utils.energy import make_meter
    from utils.env_info import environment
    device = cfg["experiment"]["device"]
    print(json.dumps(environment(cfg), indent=1))
    meter = make_meter(cfg, Path(cfg["experiment"]["results_dir"]) / f"check__{cfg['energy']['backend']}")

    def load(seconds=10):
        x, start = torch.randn(2048, 2048, device=device), time.time()
        while time.time() - start < seconds:
            x = (x @ x).tanh()
        sync(device)

    for label, work in (("idle", lambda: time.sleep(10)), ("load", load)):
        meter.begin(label)
        work()
        r = meter.end()
        watts = {k[:-2]: round(r[k] / r["duration_s"], 1) for k in ("cpu_j", "ram_j", "gpu_j", "total_j")}
        print(f"{label:5s} W={watts} measured={r['measured']}")
    meter.close()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_run":
        from train.trainer import train_run
        return train_run(sys.argv[2])
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
