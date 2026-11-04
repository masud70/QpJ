import time

import torch

from data.celeba_loader import to_float, to_uint8
from utils.display import bar


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def generate(method, n, sampler, batch_size, desc=None):
    chunks = []
    with bar(total=n, show=desc is not None, desc=desc, unit="img", leave=False) as pbar:
        for i in range(0, n, batch_size):
            k = min(batch_size, n - i)
            chunks.append(to_uint8(method.sample(k, sampler, progress=desc is not None)).cpu())
            pbar.update(k)
    return torch.cat(chunks)


def sample_for_benchmark(method, sampler, cfg):
    """Sample until both inference_images and inference_min_s are reached, so meter overhead stays negligible."""
    ev = cfg["evaluation"]
    n, start = 0, time.perf_counter()
    with bar(desc=f"Inference [{sampler}]", unit="img", leave=False) as pbar:
        while n < ev["inference_images"] or time.perf_counter() - start < ev["inference_min_s"]:
            method.sample(ev["batch_size"], sampler, progress=True)
            n += ev["batch_size"]
            pbar.update(ev["batch_size"])
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
