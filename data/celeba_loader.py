from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SPLITS = {"train": "0", "valid": "1", "test": "2"}


def to_float(images):
    return images.float() / 127.5 - 1


def to_uint8(images):
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def _load_image(path, crop, size):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BILINEAR)
    return np.asarray(img).transpose(2, 0, 1)


def _build_celeba(root, split, crop, size):
    rows = (Path(root) / "list_eval_partition.txt").read_text().split("\n")
    names = [r.split()[0] for r in rows if r.strip() and r.split()[1] == SPLITS[split]]
    paths = [Path(root) / "img_align_celeba" / n for n in names]
    with ProcessPoolExecutor() as pool:
        arrays = list(pool.map(partial(_load_image, crop=crop, size=size), paths, chunksize=512))
    return torch.from_numpy(np.stack(arrays))


def _synthetic(n, size):
    gen = torch.Generator().manual_seed(0)
    low = torch.rand(n, 3, 4, 4, generator=gen) * 2 - 1
    return to_uint8(F.interpolate(low, size=size, mode="bilinear", align_corners=False))


def load_images(cfg, split="train"):
    data = cfg["data"]
    if data["source"] == "synthetic":
        return _synthetic(data["synthetic_size"], data["image_size"])
    cache = Path(data["cache_dir"]) / f"celeba_{split}_{data['image_size']}.pt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_build_celeba(data["root"], split, data["center_crop"], data["image_size"]), cache)
    return torch.load(cache)


class Batches:
    def __init__(self, images, batch_size, device, seed):
        self.images, self.batch_size, self.device = images, batch_size, device
        self.gen = torch.Generator().manual_seed(seed)
        self.order, self.pos, self.images_seen = None, len(images), 0

    def __call__(self):
        if self.pos + self.batch_size > len(self.images):
            self.order, self.pos = torch.randperm(len(self.images), generator=self.gen), 0
        idx = self.order[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        self.images_seen += self.batch_size
        return to_float(self.images[idx].to(self.device))

    @property
    def epochs(self):
        return self.images_seen / len(self.images)
