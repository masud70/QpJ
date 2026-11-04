import time
import torch
from data.celeba_loader import to_float, to_uint8


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def generate(method, n, sampler, batch_size):
    return torch.cat([to_uint8(method.sample(min(batch_size, n - i), sampler)).cpu()
                      for i in range(0, n, batch_size)])


def sample_for_benchmark(method, sampler, cfg):
    """Sample until both inference_images and inference_min_s are reached, so meter overhead stays negligible."""
    ev = cfg["evaluation"]
    n, start = 0, time.perf_counter()
    while n < ev["inference_images"] or time.perf_counter() - start < ev["inference_min_s"]:
        method.sample(ev["batch_size"], sampler)
        n += ev["batch_size"]
    return n


@torch.no_grad()
def recon_mse(method, images, batch_size):
    total = 0.0
    for b in images.split(batch_size):
        x = to_float(b.to(method.device))
        recon = method.reconstruct(x)
        if recon is None:
            return None
        total += ((recon - x) ** 2).mean().item() * len(b)
    return total / len(images)
