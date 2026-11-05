import json
from pathlib import Path

from utils.env_info import ram_gb

J_PER_KWH = 3.6e6


class Meter:
    """Segment-based energy meter. Subclasses implement _begin/_end/_close.

    _end returns cpu_j, gpu_j, duration_s, measured flags, and ram_j (None if RAM has no counter)."""

    def __init__(self, cfg, raw_dir):
        self.cfg, self.ecfg = cfg, cfg["energy"]
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.segments, self.label, self.closed = [], None, False

    def begin(self, label):
        self.label = label
        self._begin(label)

    def end(self):
        r = self._end()
        if r["ram_j"] is None:
            r["ram_j"] = self.ecfg["ram_w_per_gb"] * ram_gb() * r["duration_s"]
            r["measured"]["ram"] = False
        r["total_j"] = r["cpu_j"] + r["ram_j"] + r["gpu_j"]
        self.segments.append({"label": self.label, **r})
        return {k: r[k] for k in ("cpu_j", "ram_j", "gpu_j", "total_j", "duration_s", "measured")}

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._close()
        (self.raw_dir / "segments.json").write_text(json.dumps(self.segments, indent=1))


def make_meter(cfg, raw_dir):
    backend = cfg["energy"]["backend"]
    if backend == "native":
        from utils.energy.native import NativeMeter as cls
    elif backend == "codecarbon":
        from utils.energy.codecarbon_meter import CodeCarbonMeter as cls
    else:
        from utils.energy.codegreen_meter import CodeGreenMeter as cls
    return cls(cfg, raw_dir)
