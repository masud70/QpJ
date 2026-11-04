from models.backbone import Encoder, Generator


class Discriminator(Encoder):
    def __init__(self, ndf, image_size):
        super().__init__(1, ndf, image_size, norm="batch")

    def forward(self, x):
        return super().forward(x).view(-1)


__all__ = ["Generator", "Discriminator"]
