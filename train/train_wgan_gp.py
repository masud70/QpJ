import torch

from models.wgan_gp import Critic, Generator
from train.base import Method, adam, step


def gradient_penalty(critic, real, fake):
    eps = torch.rand(len(real), 1, 1, 1, device=real.device)
    mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(mix).sum(), mix, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


class WGANGP(Method):
    """Wasserstein loss with gradient penalty; n_critic fresh batches per generator step (Gulrajani et al., 2017)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        m, self.c = cfg["model"], cfg["wgan_gp"]
        self.g = Generator(self.latent, m["ngf"], self.size).to(device)
        self.critic = Critic(m["ndf"], self.size).to(device)
        self.optimizers = [adam(self.g.parameters(), self.c), adam(self.critic.parameters(), self.c)]
        self.batches_per_step = self.c["n_critic"]

    def networks(self):
        return {"generator": self.g, "critic": self.critic}

    def train_step(self, next_batch):
        for _ in range(self.c["n_critic"]):
            x = next_batch()
            with torch.no_grad():
                fake = self.g(self.noise(len(x)))
            loss_c = self.critic(fake).mean() - self.critic(x).mean()
            gp = gradient_penalty(self.critic, x, fake)
            step(self.optimizers[1], loss_c + self.c["lambda_gp"] * gp)
        loss_g = -self.critic(self.g(self.noise(len(x)))).mean()
        step(self.optimizers[0], loss_g)
        return {"loss_g": loss_g.detach(), "loss_c": loss_c.detach(), "gp": gp.detach()}

    @torch.no_grad()
    def sample(self, n, sampler="direct", progress=False):
        return self.g(self.noise(n))
