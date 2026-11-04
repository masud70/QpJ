from models.backbone import Encoder, Generator


class Critic(Encoder):
    """Layer norm instead of batch norm: the gradient penalty is per-sample (Gulrajani et al., 2017)."""

    def __init__(self, ndf, image_size):
        super().__init__(1, ndf, image_size, norm="layer")

    def forward(self, x):
        return super().forward(x).view(-1)


__all__ = ["Generator", "Critic"]
