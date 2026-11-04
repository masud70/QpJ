import torch
import torch.nn.functional as F

from models.dcgan import Discriminator, Generator
from train.base import Method, adam, step


class DCGAN(Method):
    """Non-saturating GAN loss with Radford et al. (2016) architecture and optimizer."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        m, c = cfg["model"], cfg["dcgan"]
        self.g = Generator(self.latent, m["ngf"], self.size).to(device)
        self.d = Discriminator(m["ndf"], self.size).to(device)
        self.optimizers = [adam(self.g.parameters(), c), adam(self.d.parameters(), c)]

    def networks(self):
        return {"generator": self.g, "discriminator": self.d}

    def train_step(self, next_batch):
        x = next_batch()
        fake = self.g(self.noise(len(x)))
        real_logits, fake_logits = self.d(x), self.d(fake.detach())
        loss_d = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        step(self.optimizers[1], loss_d)
        gen_logits = self.d(fake)
        loss_g = F.binary_cross_entropy_with_logits(gen_logits, torch.ones_like(gen_logits))
        step(self.optimizers[0], loss_g)
        return {"loss_g": loss_g.detach(), "loss_d": loss_d.detach()}

    @torch.no_grad()
    def sample(self, n, sampler="direct", progress=False):
        return self.g(self.noise(n))
