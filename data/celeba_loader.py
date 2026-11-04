import io
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from data.download import download


def to_float(images):
    return images.float() / 127.5 - 1


def to_uint8(images):
    return ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def _load_image(data, crop, size):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BILINEAR)
    return np.asarray(img).transpose(2, 0, 1)


def _build_celeba(data, split):
    download(data["root"], data["hf_revision"])
    arrays, load = [], partial(_load_image, crop=data["center_crop"], size=data["image_size"])
    with ProcessPoolExecutor() as pool:
        for shard in sorted((Path(data["root"]) / "parquet").glob(f"{split}-*.parquet")):
            images = pq.read_table(shard, columns=["image"]).column("image").to_pylist()
            arrays += pool.map(load, [img["bytes"] for img in images], chunksize=256)
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
        torch.save(_build_celeba(data, split), cache)
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
