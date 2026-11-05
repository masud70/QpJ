"""Console output: messages go to the terminal (and a log file), progress bars only to an interactive terminal."""
import logging
import sys
import textwrap
import time

from tqdm import tqdm

NAMES = {"ae": "AE", "vae": "VAE", "dcgan": "DCGAN", "wgan_gp": "WGAN-GP", "ddpm": "DDPM"}
log = logging.getLogger("qpj")


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record), file=sys.stdout)
        sys.stdout.flush()


def setup_logging(log_file=None):
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    log.addHandler(_TqdmHandler())
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)


def bar(iterable=None, show=True, **kwargs):
    return tqdm(iterable, file=sys.stdout, disable=None if show else True, dynamic_ncols=True, mininterval=0.1,
                **kwargs)


def countdown(seconds, desc):
    for _ in bar(range(int(seconds)), desc=desc, leave=False, bar_format="{desc}: {bar} {n_fmt}/{total_fmt} s"):
        time.sleep(1)


def row(label, text, width=100):
    lines = textwrap.wrap(str(text), width - 11, break_on_hyphens=False, break_long_words=False) or [""]
    log.info(f"{label:<10} {lines[0]}")
    for line in lines[1:]:
        log.info(f"{'':<11}{line}")


def si(n, unit=""):
    for suffix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.1f}{' ' if unit else ''}{suffix}{unit}"
    return f"{n:g}{' ' if unit else ''}{unit}"


def joules(j):
    for unit, scale in (("MJ", 1e6), ("kJ", 1e3)):
        if abs(j) >= scale:
            return f"{j / scale:,.1f} {unit}"
    return f"{j:,.1f} J"


def energy_text(r):
    return f"{joules(r['total_j'])} (CPU {joules(r['cpu_j'])} · RAM {joules(r['ram_j'])} · GPU {joules(r['gpu_j'])})"


def power_text(watts, measured):
    parts = [f"{c.upper()} {watts[c]:.1f} W {'✓' if measured[c] else '✗'}" for c in ("cpu", "ram", "gpu")]
    return " · ".join(parts) + "   (✓ measured, ✗ estimated)"


def budget_text(b):
    return f"{b['value']:g} {b['unit']}"


def stopping_text(cfg):
    s, mode = cfg["stopping"], cfg["stopping"]["mode"]
    if mode == "fixed_budget":
        return f"fixed_budget · {budget_text(s[mode]['budget'])} · segments of {budget_text(s[mode]['segment'])}"
    tq = s[mode]
    return (f"target_quality · FID ≤ {tq['target_fid']:g} · checked every {budget_text(tq['eval_every'])} "
            f"on {tq['fid_samples']:,} images · cap {budget_text(tq['max_budget'])}")


def config_text(block):
    fmt = lambda v: f"({', '.join(map(str, v))})" if isinstance(v, list) else f"{v:g}" if isinstance(v, float) else v
    return " · ".join(f"{k} {fmt(v)}" for k, v in block.items())
