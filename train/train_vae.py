import torch

from models.vae import VAE as VAENet
from train.base import Method, adam, step


class VAE(Method):
    """ELBO summed per image: Gaussian decoder with fixed variance plus analytic KL (Kingma & Welling, 2014)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        m, self.c = cfg["model"], cfg["vae"]
        self.net = VAENet(self.latent, m["ngf"], m["ndf"], self.size).to(device)
        self.optimizers = [adam(self.net.parameters(), self.c)]

    def networks(self):
        return {"vae": self.net}

    def train_step(self, next_batch):
        x = next_batch()
        recon, mu, logvar = self.net(x)
        rec = ((recon - x) ** 2).sum() / (2 * self.c["decoder_var"] * len(x))
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum() / len(x)
        loss = rec + self.c["beta"] * kl
        step(self.optimizers[0], loss)
        return {"loss": loss.detach(), "recon": rec.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def reconstruct(self, x):
        return self.net.decoder(self.net.encoder(x).chunk(2, dim=1)[0])

    @torch.no_grad()
    def sample(self, n, sampler="direct", progress=False):
        return self.net.decoder(self.noise(n))
