"""Download CelebA (official aligned & cropped images, PNG in Parquet) from the Hugging Face mirror flwrlabs/celeba."""
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve

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
    for path, size in shards(revision):
        dest = out / Path(path).name
        if dest.exists() and dest.stat().st_size == size:
            continue
        print(f"Downloading {dest.name} ({size / 1e6:.0f} MB)", flush=True)
        part = dest.with_suffix(".part")
        urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{quote(path)}", part)
        if part.stat().st_size != size:
            raise RuntimeError(f"{dest.name}: expected {size} bytes, got {part.stat().st_size}")
        part.rename(dest)
