from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "default.yaml"
PATH_KEYS = [("experiment", "results_dir"), ("data", "root"), ("data", "cache_dir")]
CHOICES = {
    ("energy", "backend"): {"native", "codecarbon", "codegreen"},
    ("stopping", "mode"): {"fixed_budget", "target_quality"},
    ("data", "source"): {"celeba", "synthetic"},
}
MODELS = {"ae", "vae", "dcgan", "wgan_gp", "ddpm"}
UNITS = {"epochs", "steps", "images", "wall_clock_s", "energy_j"}


def merge(base, new):
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def set_key(cfg, dotted, value):
    *parents, key = dotted.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if key not in node:
        raise KeyError(f"Unknown config key: {dotted}")
    node[key] = yaml.safe_load(value)


def validate(cfg):
    for (section, key), allowed in CHOICES.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f"{section}.{key} must be one of {sorted(allowed)}")
    if not set(cfg["experiment"]["models"]) <= MODELS:
        raise ValueError(f"experiment.models must be a subset of {sorted(MODELS)}")
    stop = cfg["stopping"]
    budgets = [stop["fixed_budget"]["budget"], stop["fixed_budget"]["segment"],
               stop["target_quality"]["eval_every"], stop["target_quality"]["max_budget"]]
    for b in budgets:
        if b["unit"] not in UNITS:
            raise ValueError(f"Budget unit must be one of {sorted(UNITS)}")
    for b in budgets[1:3]:
        if b["unit"] == "energy_j":
            raise ValueError("Segment and eval_every units cannot be energy_j")
    ddim = cfg["ddpm"]["ddim_steps"]
    if ddim is not None and not (isinstance(ddim, int) and 1 <= ddim <= cfg["ddpm"]["T"]):
        raise ValueError("ddpm.ddim_steps must be null or an integer in [1, T]")
    if cfg["energy"]["native"]["poll_s"] < 0.05:
        raise ValueError("energy.native.poll_s must be >= 0.05 s")


def load_config(path=None, overrides=()):
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    if path:
        merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides:
        set_key(cfg, *item.split("=", 1))
    for section, key in PATH_KEYS:
        cfg[section][key] = str((ROOT / cfg[section][key]).resolve())
    validate(cfg)
    return cfg
