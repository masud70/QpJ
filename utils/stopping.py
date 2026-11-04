class Progress:
    def __init__(self, batches):
        self.batches = batches
        self.steps, self.train_time_s, self.energy_j = 0, 0.0, 0.0

    def value(self, unit):
        return {
            "epochs": self.batches.epochs,
            "steps": self.steps,
            "images": self.batches.images_seen,
            "wall_clock_s": self.train_time_s,
            "energy_j": self.energy_j,
        }[unit]

    def as_dict(self):
        return {"steps": self.steps, "images_seen": self.batches.images_seen,
                "epochs": round(self.batches.epochs, 4), "train_time_s": round(self.train_time_s, 2)}


class Stopper:
    """Training runs in segments; energy_j budgets are checked at segment ends (overshoot <= one segment)."""

    def __init__(self, cfg):
        s = cfg["stopping"]
        self.mode = s["mode"]
        if self.mode == "fixed_budget":
            self.budget, self.segment, self.target = s["fixed_budget"]["budget"], s["fixed_budget"]["segment"], None
        else:
            tq = s["target_quality"]
            self.budget, self.segment, self.target = tq["max_budget"], tq["eval_every"], tq["target_fid"]
        self.boundary = self.segment["value"]
        self.status = None

    @property
    def evaluates(self):
        return self.target is not None

    def budget_done(self, p):
        return p.value(self.budget["unit"]) >= self.budget["value"]

    def segment_done(self, p):
        return p.value(self.segment["unit"]) >= self.boundary or self.budget_done(p)

    def end_segment(self, p, fid=None):
        while p.value(self.segment["unit"]) >= self.boundary:
            self.boundary += self.segment["value"]
        if fid is not None and fid <= self.target:
            self.status = "target_reached"
        elif self.budget_done(p):
            self.status = "completed" if self.mode == "fixed_budget" else "budget_exhausted"
        return self.status is not None
