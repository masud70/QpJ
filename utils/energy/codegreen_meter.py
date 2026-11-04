import codegreen

from utils.energy import Meter

CODEGREEN_HINT = "CodeGreen sensors unavailable. Run 'sudo codegreen init-sensors' and retry."


def keys(domains, *prefixes):
    return [k for k in domains if k.lower().startswith(prefixes)]


class CodeGreenMeter(Meter):
    """CodeGreen session tasks. CPU = top-level package domains, RAM = dram, GPU = NVML/IOReport gpu domains."""

    def __init__(self, cfg, raw_dir):
        super().__init__(cfg, raw_dir)
        self.session = codegreen.Session("qpj", output_file=str(self.raw_dir / "codegreen.json"),
                                         sample_interval_ms=self.ecfg["codegreen"]["sample_interval_ms"]).start()
        if self.session._noop:
            raise RuntimeError(CODEGREEN_HINT)

    def _begin(self, label):
        self.session.start_task(label)

    def _end(self):
        r = self.session.stop_task()
        d = r.domains
        cpu = keys(d, "package", "pkg") or keys(d, "cpu_energy")
        gpu = ["gpu_energy"] if "gpu_energy" in d else keys(d, "gpu")
        ram = keys(d, "dram")
        return {
            "cpu_j": sum(d[k] for k in cpu),
            "ram_j": sum(d[k] for k in ram) if ram else None,
            "gpu_j": sum(d[k] for k in gpu),
            "duration_s": r.duration_s,
            "measured": {"cpu": bool(cpu), "ram": bool(ram), "gpu": bool(gpu)},
            "domains": d,
        }

    def _close(self):
        self.session.stop()
