import torch.nn as nn

from models.backbone import Encoder, Generator


class Autoencoder(nn.Module):
    def __init__(self, latent_dim, ngf, ndf, image_size):
        super().__init__()
        self.encoder = Encoder(latent_dim, ndf, image_size)
        self.decoder = Generator(latent_dim, ngf, image_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))
