from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
IMAGES = DATA / "images"
CACHE = DATA / "cache" / "img256"

LABELS = {"true": 0, "out-of-context": 1, "miscaptioned": 2}
LABEL_NAMES = ["true", "out-of-context", "miscaptioned"]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_name(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:24] + ".jpg"


def load_pairs(path: Path | None = None) -> list[dict]:
    path = path or DATA / "07_pairs.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_caption(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def letterbox(im: Image.Image, size: int) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def cache_path(url: str) -> Path:
    return CACHE / image_name(url)


def _cache_one(payload: tuple[str, int, int]) -> None:
    url, size, quality = payload
    with Image.open(IMAGES / image_name(url)) as im:
        letterbox(im, size).save(cache_path(url), "JPEG", quality=quality, optimize=True)


def build_cache(size: int = 256, quality: int = 92, workers: int = 4) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    urls = sorted({r["image"] for r in load_pairs()})
    todo = [u for u in urls if not cache_path(u).exists()]
    if not todo:
        return 0

    jobs = [(url, size, quality) for url in todo]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            pool.map(_cache_one, jobs, chunksize=16)
    else:
        for job in jobs:
            _cache_one(job)
    return len(todo)


class PairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_len: int = 128,
        image_size: int = 224,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size
        self.captions = [clean_caption(r["caption"]) for r in rows]
        self.paths = [cache_path(r["image"]) for r in rows]
        self.labels = [LABELS[r["label"]] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            self.captions[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if im.size[0] != self.image_size:
                im = im.resize((self.image_size, self.image_size), Image.BICUBIC)
            pixel = torch.from_numpy(np.asarray(im, dtype="uint8").copy())
        pixel = pixel.permute(2, 0, 1).float().div_(255.0)
        pixel = (pixel - MEAN) / STD
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": pixel,
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }
