import copy

import torch

from models.ddpm import Diffusion, UNet
from train.base import Method, adam


class DDPM(Method):
    """Ho et al. (2020): epsilon-prediction, LR warmup, gradient clipping, EMA weights for sampling."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.c = c = cfg["ddpm"]
        self.net = UNet(c["ch"], c["ch_mult"], c["num_res_blocks"], c["attn_resolutions"], c["dropout"],
                        self.size).to(device)
        self.ema = copy.deepcopy(self.net).eval().requires_grad_(False)
        self.diffusion = Diffusion(c["T"], c["beta_start"], c["beta_end"], device)
        opt = adam(self.net.parameters(), c)
        self.optimizers = [opt]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c["warmup_steps"]))
        self.samplers = [c["final_sampler"]] + [s for s in c["inference_samplers"] if s != c["final_sampler"]]
        self.eval_sampler = c["eval_sampler"]

    def networks(self):
        return {"unet": self.net}

    def state(self):
        return {"unet": self.net.state_dict(), "ema": self.ema.state_dict()}

    def network_calls(self, sampler):
        return self.diffusion.steps(sampler)

    def train_step(self, next_batch):
        loss = self.diffusion.loss(self.net, next_batch())
        opt = self.optimizers[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.c["grad_clip"])
        opt.step()
        self.scheduler.step()
        with torch.no_grad():
            for e, p in zip(self.ema.parameters(), self.net.parameters()):
                e.lerp_(p, 1 - self.c["ema_decay"])
        return {"loss": loss.detach()}

    def sample(self, n, sampler):
        return self.diffusion.sample(self.ema, n, self.size, self.device, sampler)
