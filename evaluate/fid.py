from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data.celeba_loader import load_images


class FID:
    """Standard FID Inception-v3 (torch-fidelity weights via torchmetrics) on uint8 images."""

    def __init__(self, device, batch_size):
        self.net = FrechetInceptionDistance(feature=2048).inception.to(device).eval()
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def stats(self, images):
        feats = torch.cat([self.net(b.to(self.device)).double().cpu() for b in images.split(self.batch_size)])
        return feats.mean(0), torch.cov(feats.T)

    def score(self, images, ref):
        mu1, s1 = self.stats(images)
        mu2, s2 = ref
        covmean_trace = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum()
        return float(((mu1 - mu2) ** 2).sum() + s1.trace() + s2.trace() - 2 * covmean_trace)


def reference_stats(cfg, fid):
    data, n = cfg["data"], cfg["evaluation"]["fid_ref_samples"]
    path = Path(data["cache_dir"]) / f"fid_ref_{data['source']}_train_{data['image_size']}_{n}.pt"
    if not path.exists():
        images = load_images(cfg, "train")
        idx = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[:n]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fid.stats(images[idx]), path)
    return torch.load(path)
