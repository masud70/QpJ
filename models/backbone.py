import math

import torch.nn as nn

NORMS = {
    "batch": nn.BatchNorm2d,
    "layer": lambda ch: nn.GroupNorm(1, ch),
    "none": lambda ch: nn.Identity(),
}


def widths(base, image_size):
    return [base * 2 ** i for i in range(int(math.log2(image_size // 4)))]


def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """Radford et al. (2016) generator: z -> 4x4 -> ... -> image."""

    def __init__(self, latent_dim, ngf, image_size):
        super().__init__()
        w = widths(ngf, image_size)[::-1]
        layers = [nn.ConvTranspose2d(latent_dim, w[0], 4, 1, 0, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(True)]
        for cin, cout in zip(w, w[1:]):
            layers += [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(w[-1], 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, z):
        return self.net(z.view(*z.shape, 1, 1))


class Encoder(nn.Module):
    """Radford et al. (2016) discriminator body with a configurable head and norm."""

    def __init__(self, out_dim, ndf, image_size, norm="batch"):
        super().__init__()
        w = widths(ndf, image_size)
        bias = norm == "none"
        layers = [nn.Conv2d(3, w[0], 4, 2, 1, bias=bias), nn.LeakyReLU(0.2, True)]
        for cin, cout in zip(w, w[1:]):
            layers += [nn.Conv2d(cin, cout, 4, 2, 1, bias=bias), NORMS[norm](cout), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(w[-1], out_dim, 4, 1, 0)]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, x):
        return self.net(x).flatten(1)
