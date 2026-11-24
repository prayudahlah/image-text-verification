from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .data import DATA, IMAGES, image_name, load_pairs

ARSIP = DATA / "arsip"
CONFIG = Path(__file__).resolve().parents[1] / "arsip.json"
MANIFEST = DATA / "04_images.jsonl"


def expected_images() -> dict[str, int]:
    """Nama berkas yang dipakai dataset, beserta ukuran byte yang benar."""
    if MANIFEST.exists():
        out = {}
        with MANIFEST.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "ok" and rec.get("file"):
                    out[rec["file"]] = int(rec["bytes"])
        if out:
            return out
    out = {}
    for row in load_pairs():
        name = image_name(row["image"])
        path = IMAGES / name
        if path.exists():
            out[name] = path.stat().st_size
    return out


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_config(path: Path | None = None) -> dict:
    path = path or CONFIG
    if not path.exists():
        raise FileNotFoundError(f"konfigurasi arsip tidak ada: {path}")
    return json.loads(path.read_text())


def tar_members(path: Path) -> list[str]:
    with tarfile.open(path) as tar:
        return [m.name for m in tar.getmembers() if m.isfile()]


def missing_images(expected: dict[str, int]) -> list[str]:
    missing = []
    for name, size in expected.items():
        path = IMAGES / name
        if not path.exists() or path.stat().st_size != size:
            missing.append(name)
    return missing


def ensure_images(download: bool = True, config_path: Path | None = None) -> dict:
    """Pastikan semua gambar dataset ada. Idempoten, aman dijalankan berulang."""
    expected = expected_images()
    missing = missing_images(expected)
    if not missing:
        return {"status": "lengkap", "jumlah": len(expected)}

    cfg = read_config(config_path)
    need = set(missing)
    dipulihkan: list[str] = []
    diunduh: list[str] = []

    for part in cfg["bagian"]:
        tar_path = ARSIP / part["nama"]

        if tar_path.exists() and sha256(tar_path) != part["sha256"]:
            tar_path.unlink()

        if not tar_path.exists():
            if not download:
                raise RuntimeError(
                    f"{tar_path.name} tidak ada dan unduhan dimatikan. "
                    f"Taruh arsipnya di {ARSIP}."
                )
            drive_id = (part.get("drive_id") or "").strip()
            if not drive_id:
                raise RuntimeError(
                    f"drive_id untuk {part['nama']} belum diisi di {CONFIG}. "
                    "Isi setelah arsip diunggah ke Drive."
                )
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError(
                    "gdown belum terpasang. Jalankan: "
                    "uv pip install --python .venv-train/bin/python gdown"
                ) from exc
            ARSIP.mkdir(parents=True, exist_ok=True)
            gdown.download(id=drive_id, output=str(tar_path), quiet=False)
            got = sha256(tar_path)
            if got != part["sha256"]:
                tar_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"sha256 {tar_path.name} tidak cocok. "
                    f"harap {part['sha256']}, dapat {got}. Coba unduh ulang."
                )
            diunduh.append(part["nama"])

        members = [m for m in tar_members(tar_path) if m in need]
        if not members:
            continue
        with tarfile.open(tar_path) as tar:
            for name in members:
                tar.extract(name, path=IMAGES, filter="data")
                dipulihkan.append(name)
                need.discard(name)

    sisa = missing_images(expected)
    return {
        "status": "lengkap" if not sisa else "kurang",
        "jumlah": len(expected),
        "hilang_awal": len(missing),
        "dipulihkan": len(dipulihkan),
        "diunduh": diunduh,
        "masih_hilang": len(sisa),
        "contoh_hilang": sisa[:5],
    }
