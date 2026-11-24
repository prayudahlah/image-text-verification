from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.csv"

FIELDS = [
    "run_id", "kode", "faktor", "teks", "gambar", "fusion",
    "lr", "dropout", "batch", "seed", "max_epochs", "patience",
    "epochs_run", "epoch_best",
    "acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc",
    "val_n", "wall_clock_s", "params", "vram_peak_mb", "catatan", "dibuat",
]


def score(y_true, y_pred) -> dict:
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_true": float(f1[0]),
        "f1_ooc": float(f1[1]),
        "f1_mc": float(f1[2]),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def read_ledger(path: Path | None = None) -> list[dict]:
    path = path or LEDGER
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def done_ids(path: Path | None = None) -> set[str]:
    return {row["run_id"] for row in read_ledger(path)}


def append_ledger(row: dict, path: Path | None = None) -> None:
    path = path or LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("dibuat", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def top2_gap(rows: list[dict]) -> tuple[float, str, str]:
    ranked = sorted(rows, key=lambda r: float(r["f1_macro"]), reverse=True)
    if len(ranked) < 2:
        return 0.0, ranked[0]["run_id"], ""
    return (
        float(ranked[0]["f1_macro"]) - float(ranked[1]["f1_macro"]),
        ranked[0]["run_id"],
        ranked[1]["run_id"],
    )
