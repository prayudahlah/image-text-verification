from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Run:
    kode: str
    faktor: str
    teks: str | None
    gambar: str | None
    fusion: str
    lr: float = 2e-5
    dropout: float = 0.1
    batch: int = 16
    seed: int = 727
    max_epochs: int = 8
    patience: int = 2

    @property
    def run_id(self) -> str:
        t = self.teks or "na"
        g = self.gambar or "na"
        return f"{self.kode}_{t}_{g}_{self.fusion}_s{self.seed}"

    def row(self) -> dict:
        return asdict(self) | {"run_id": self.run_id}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)


def loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
