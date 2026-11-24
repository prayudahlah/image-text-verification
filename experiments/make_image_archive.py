from __future__ import annotations

import json
import tarfile

from experiments.lib.data import IMAGES
from experiments.lib.images import ARSIP, CONFIG, expected_images, sha256, tar_members

PARTS = 5
BASENAME = "gambar-13674"


def main() -> int:
    expected = expected_images()
    names = sorted(expected)
    ARSIP.mkdir(parents=True, exist_ok=True)

    lama = {}
    if CONFIG.exists():
        lama = {p["nama"]: p for p in json.loads(CONFIG.read_text()).get("bagian", [])}

    bagian = []
    for i in range(PARTS):
        chunk = names[i * len(names) // PARTS : (i + 1) * len(names) // PARTS]
        tar_path = ARSIP / f"{BASENAME}.part{i + 1}.tar"

        if tar_path.exists():
            isi = set(tar_members(tar_path))
            if isi == set(chunk):
                print(f"lewati {tar_path.name}, isinya sudah cocok")
            else:
                tar_path.unlink()

        if not tar_path.exists():
            with tarfile.open(tar_path, "w") as tar:
                for name in chunk:
                    tar.add(IMAGES / name, arcname=name)
            print(f"tulis {tar_path.name} ({len(chunk)} berkas)")

        digest = sha256(tar_path)
        rec = lama.get(tar_path.name, {})
        drive_id = rec.get("drive_id", "") if rec.get("sha256") == digest else ""
        bagian.append(
            {
                "nama": tar_path.name,
                "berkas": f"{i + 1}/{PARTS}",
                "jumlah": len(chunk),
                "bytes": tar_path.stat().st_size,
                "sha256": digest,
                "drive_id": drive_id,
            }
        )

    total = sum(p["bytes"] for p in bagian)
    cfg = {
        "dataset": "data/07_pairs.jsonl",
        "jumlah_berkas": len(names),
        "total_bytes": total,
        "bagian": bagian,
    }
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"\narsip: {ARSIP}")
    print(f"total: {total / 1e6:.0f} MB dalam {PARTS} bagian")
    for p in bagian:
        print(f"  {p['nama']:28s} {p['bytes'] / 1e6:6.0f} MB  {p['sha256'][:16]}...")
    print(f"\nkonfigurasi: {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
