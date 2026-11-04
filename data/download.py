"""Download CelebA (official aligned & cropped images, PNG in Parquet) from the Hugging Face mirror flwrlabs/celeba."""
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

from utils.display import bar, log

REPO = "flwrlabs/celeba"
SUBSET = "img_align+identity+attr"
SPLITS = ("train", "test")


def shards(revision):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{quote(SUBSET)}"
    with urlopen(url) as response:
        files = json.load(response)
    return [(f["path"], f["size"]) for f in files if Path(f["path"]).name.startswith(SPLITS)]


def download(root, revision):
    out = Path(root) / "parquet"
    out.mkdir(parents=True, exist_ok=True)
    missing = [(p, s) for p, s in shards(revision)
               if not ((out / Path(p).name).exists() and (out / Path(p).name).stat().st_size == s)]
    if missing:
        log.info(f"Downloading CelebA from Hugging Face ({REPO} @ {revision[:7]}): {len(missing)} files, "
                 f"{sum(s for _, s in missing) / 1e9:.1f} GB → {out}")
    for i, (path, size) in enumerate(missing, 1):
        dest, part = out / Path(path).name, out / (Path(path).name + ".part")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(path)}"
        with bar(total=size, desc=f"[{i}/{len(missing)}] {dest.name}", unit="B", unit_scale=True, leave=False) as pbar:
            urlretrieve(url, part, reporthook=lambda blocks, block_size, _: pbar.update(blocks * block_size - pbar.n))
        if part.stat().st_size != size:
            raise RuntimeError(f"{dest.name}: expected {size} bytes, got {part.stat().st_size}")
        part.rename(dest)
