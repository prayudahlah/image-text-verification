from __future__ import annotations

import os
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / (os.getenv("NOTEBOOK") or "03_eksperimen_fusion_v2.py")
DST = SRC.with_suffix(".ipynb")
KERNEL = "image-text-fusion"


def main() -> int:
    nb = jupytext.read(SRC)
    nb.metadata["kernelspec"] = {
        "name": KERNEL,
        "display_name": KERNEL,
        "language": "python",
    }
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    nb = client.execute()
    nbformat.write(nb, DST)
    print("selesai:", DST.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
