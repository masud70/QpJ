from train.train_ae import AE
from train.train_dcgan import DCGAN
from train.train_ddpm import DDPM
from train.train_vae import VAE
from train.train_wgan_gp import WGANGP

METHODS = {"ae": AE, "vae": VAE, "dcgan": DCGAN, "wgan_gp": WGANGP, "ddpm": DDPM}


def build_method(name, cfg, device):
    return METHODS[name](cfg, device)
