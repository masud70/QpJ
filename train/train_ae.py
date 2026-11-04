import torch

from data.celeba_loader import to_float
from models.autoencoder import Autoencoder
from train.base import Method, adam, step


class AE(Method):
    """Deterministic AE; samples from a full-covariance Gaussian fitted to training latents (Ghosh et al., 2020)."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        m = cfg["model"]
        self.net = Autoencoder(self.latent, m["ngf"], m["ndf"], self.size).to(device)
        self.optimizers = [adam(self.net.parameters(), cfg["ae"])]
        self.prior = None

    def networks(self):
        return {"autoencoder": self.net}

    def train_step(self, next_batch):
        x = next_batch()
        loss = ((self.net(x) - x) ** 2).sum() / len(x)
        step(self.optimizers[0], loss)
        return {"loss": loss.detach()}

    @torch.no_grad()
    def prepare_sampling(self, images):
        idx = torch.randperm(len(images))[:self.cfg["ae"]["fit_samples"]]
        z = torch.cat([self.net.encoder(to_float(images[i].to(self.device))) for i in idx.split(500)])
        cov = torch.cov(z.T) + 1e-5 * torch.eye(z.shape[1], device=z.device)
        self.prior = torch.distributions.MultivariateNormal(z.mean(0), covariance_matrix=cov)

    @torch.no_grad()
    def reconstruct(self, x):
        return self.net(x)

    @torch.no_grad()
    def sample(self, n, sampler="direct"):
        return self.net.decoder(self.prior.sample((n,)))
