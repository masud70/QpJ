import os
import platform
import socket
import subprocess
from importlib.metadata import version

import torch

from config.config import ROOT


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=ROOT).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def require_gpu(cfg):
    if cfg["experiment"]["device"].startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("experiment.device is 'cuda' but PyTorch sees no GPU. Check the NVIDIA driver and the "
                         "PyTorch CUDA build, or run with --set experiment.device=cpu.")


def ram_gb():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3


def cpu_model():
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()


def gpu_info(index):
    if not torch.cuda.is_available():
        return {"gpu": None}
    query = run(["nvidia-smi", f"--id={index}", "--format=csv,noheader,nounits",
                 "--query-gpu=driver_version,power.limit,clocks.max.sm,temperature.gpu"])
    fields = (query or ",,,").split(", ")
    return {"gpu": torch.cuda.get_device_name(index), "driver": fields[0], "gpu_power_limit_w": fields[1],
            "gpu_max_sm_clock_mhz": fields[2], "gpu_temperature_c": fields[3]}


def gpu_summary(index):
    """GPU description via nvidia-smi only, so the calling process never creates a CUDA context."""
    query = run(["nvidia-smi", f"--id={index}", "--format=csv,noheader",
                 "--query-gpu=name,memory.total,driver_version"])
    return " · ".join(query.split(", ")[:2]) + f" · driver {query.split(', ')[2]}" if query else None


def environment(cfg):
    backend = cfg["energy"]["backend"]
    dirty = run(["git", "status", "--porcelain"])
    return {
        "host": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_model(),
        "ram_gb": round(ram_gb(), 2),
        **gpu_info(cfg["energy"]["gpu_index"]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "backend_version": None if backend == "native" else version(backend),
        "git_commit": run(["git", "rev-parse", "--verify", "-q", "HEAD"]),
        "git_dirty": bool(dirty),
    }
