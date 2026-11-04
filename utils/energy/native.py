"""Library-free meter: Linux RAPL via sysfs (CPU package, DRAM, psys) and NVIDIA NVML via ctypes (GPU)."""
import ctypes
import threading
import time
from pathlib import Path

from utils.energy import Meter

RAPL_HINT = ("RAPL counters are not readable. Run 'sudo chmod -R a+r /sys/class/powercap/intel-rapl*' "
             "(or add a udev rule) and retry.")


class Rapl:
    def __init__(self, root):
        self.zones = {}
        for zone in sorted(Path(root).glob("intel-rapl:*")):
            name = (zone / "name").read_text().strip()
            if name.startswith(("package", "dram", "psys")):
                self.zones[f"{zone.name}:{name}"] = (zone / "energy_uj", int((zone / "max_energy_range_uj").read_text()))
        if not any(":package" in key for key in self.zones):
            raise RuntimeError(f"No RAPL package domain under {root}")
        self.last = self._raw()
        self.total = dict.fromkeys(self.zones, 0.0)

    def _raw(self):
        return {key: int(path.read_text()) for key, (path, _) in self.zones.items()}

    def poll(self):
        now = self._raw()
        for key, (_, max_range) in self.zones.items():
            self.total[key] += ((now[key] - self.last[key]) % max_range) / 1e6
        self.last = now
        return dict(self.total)


class Nvml:
    def __init__(self, index):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.handle)))
        self.has_counter = self._counter() is not None
        self.integrated, self.last_t = 0.0, time.perf_counter()

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"NVML error code {code}")

    def _counter(self):
        mj = ctypes.c_ulonglong()
        ok = self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(mj)) == 0
        return mj.value / 1e3 if ok else None

    def _power_w(self):
        mw = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(mw)))
        return mw.value / 1e3

    def poll(self):
        if self.has_counter:
            return self._counter()
        now = time.perf_counter()
        self.integrated += self._power_w() * (now - self.last_t)
        self.last_t = now
        return self.integrated


class NativeMeter(Meter):
    def __init__(self, cfg, raw_dir):
        super().__init__(cfg, raw_dir)
        native = self.ecfg["native"]
        try:
            self.rapl = Rapl(native["rapl_root"])
        except PermissionError as e:
            raise PermissionError(RAPL_HINT) from e
        try:
            self.nvml = Nvml(self.ecfg["gpu_index"])
        except (OSError, RuntimeError):
            if cfg["experiment"]["device"] == "cuda":
                raise
            self.nvml = None
        self.lock, self.stopped = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(native["poll_s"],), daemon=True)
        self.thread.start()

    def _loop(self, poll_s):
        while not self.stopped.wait(poll_s):
            self._read()

    def _read(self):
        with self.lock:
            return time.perf_counter(), self.rapl.poll(), self.nvml.poll() if self.nvml else 0.0

    def _begin(self, label):
        self.start = self._read()

    def _end(self):
        if not self.thread.is_alive():
            raise RuntimeError("Native energy polling thread stopped; readings would miss counter wraparounds")
        (t0, rapl0, gpu0), (t1, rapl1, gpu1) = self.start, self._read()
        domains = {key: rapl1[key] - rapl0[key] for key in rapl1}
        dram = [v for key, v in domains.items() if ":dram" in key]
        return {
            "cpu_j": sum(v for key, v in domains.items() if ":package" in key),
            "ram_j": sum(dram) if dram else None,
            "gpu_j": gpu1 - gpu0,
            "duration_s": t1 - t0,
            "measured": {"cpu": True, "ram": bool(dram), "gpu": self.nvml is not None},
            "domains": domains,
        }

    def _close(self):
        self.stopped.set()
        self.thread.join()
