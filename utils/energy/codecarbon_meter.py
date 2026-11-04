import dataclasses
import json
import logging

from codecarbon import OfflineEmissionsTracker

from utils.energy import J_PER_KWH, Meter

MEASURED_CPU_MODES = {"intel_rapl", "intel_power_gadget", "windows_emi"}


class CodeCarbonMeter(Meter):
    """CodeCarbon tasks; its RAM figure is always a model estimate, CPU is measured only in RAPL-like modes."""

    def __init__(self, cfg, raw_dir):
        super().__init__(cfg, raw_dir)
        c = self.ecfg["codecarbon"]
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        gpu_ids = [self.ecfg["gpu_index"]] if cfg["experiment"]["device"] == "cuda" else None
        self.tracker = OfflineEmissionsTracker(
            project_name="qpj", output_dir=str(self.raw_dir), measure_power_secs=c["measure_power_secs"],
            tracking_mode=c["tracking_mode"], country_iso_code=c["country_iso_code"], region=c["region"],
            gpu_ids=gpu_ids, log_level="error", allow_multiple_runs=True)
        self.tasks = []

    def _begin(self, label):
        self.tracker.start_task(label)

    def _end(self):
        d = self.tracker.stop_task()
        self.tasks.append(dataclasses.asdict(d))
        cpu_modes = [h._mode for h in self.tracker._hardware if type(h).__name__ == "CPU"]
        has_gpu = any(type(h).__name__ == "GPU" for h in self.tracker._hardware)
        return {
            "cpu_j": d.cpu_energy * J_PER_KWH,
            "ram_j": d.ram_energy * J_PER_KWH,
            "gpu_j": d.gpu_energy * J_PER_KWH,
            "duration_s": d.duration,
            "measured": {"cpu": bool(cpu_modes) and set(cpu_modes) <= MEASURED_CPU_MODES, "ram": False, "gpu": has_gpu},
            "domains": {"cpu_modes": cpu_modes, "emissions_kg": d.emissions},
        }

    def _close(self):
        self.tracker.stop()
        (self.raw_dir / "codecarbon_tasks.json").write_text(json.dumps(self.tasks, indent=1, default=str))
