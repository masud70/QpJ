import torch
import torch.nn as nn

from models.backbone import Encoder, Generator


class VAE(nn.Module):
    def __init__(self, latent_dim, ngf, ndf, image_size):
        super().__init__()
        self.encoder = Encoder(2 * latent_dim, ndf, image_size)
        self.decoder = Generator(latent_dim, ngf, image_size)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decoder(z), mu, logvar
