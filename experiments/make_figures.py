from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIG = EXP / "figures"
LEDGER = EXP / "ledger.csv"
NB = ROOT / "notebooks" / "03_eksperimen_fusion_v2.ipynb"

FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.fontsize": 7,
    }
)

BIRU = "#3b6ea5"
MERAH = "#b5654a"
ABU = "#7a8b99"
HIJAU = "#4a8c6f"
KUNING = "#c9a227"

NUM = ["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc",
       "lr", "dropout", "epoch_best", "epochs_run", "params", "wall_clock_s"]


def load_ledger() -> pd.DataFrame:
    df = pd.read_csv(LEDGER)
    for col in NUM:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates("run_id")


def label(row) -> str:
    k = row["kode"]
    if k == "B0":
        return "B0 mayoritas"
    if k == "A1":
        return f"A1 {row['teks']}"
    if k == "A2":
        return f"A2 {row['gambar']}"
    if k.startswith("F"):
        return f"{k} {row['fusion']}"
    if k.startswith("M"):
        return f"{k} {row['teks']}+{row['gambar']}"
    if k.startswith("T"):
        return f"{k} lr={row['lr']:g} do={row['dropout']:g}"
    return k


def fig_ladder(df: pd.DataFrame) -> None:
    panels = [
        ("Ablasi modalitas dan fusion", ["B0", "A1", "A2", "F0", "F1", "F2", "F3", "F4", "F5"]),
        ("Grid encoder dan tuning", ["M1", "M2", "M3", "M4", "T1", "T2", "T3", "T4", "T5", "T6"]),
        ("OOF dan model final", ["O1", "O2", "O3", "O4", "O5", "Z1", "Z2", "Z3"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (title, order) in zip(axes, panels):
        sub = df[df.kode.isin(order)].copy()
        sub["urut"] = sub.kode.map({k: i for i, k in enumerate(order)})
        sub = sub.sort_values("urut", ascending=False)
        labels = [label(r) for _, r in sub.iterrows()]
        colors = [MERAH if r.kode == "B0" else BIRU for _, r in sub.iterrows()]
        ax.barh(labels, sub.f1_macro, color=colors)
        ax.axvline(0.1961, color=MERAH, linestyle="--", linewidth=1, alpha=0.8)
        for y, v in enumerate(sub.f1_macro):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=7)
        ax.set_xlim(0, 0.82)
        ax.set_title(title)
        ax.set_xlabel("macro-F1")
    fig.suptitle("Tangga hasil dari baseline sampai model final", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "13_ladder.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_class(df: pd.DataFrame) -> None:
    z = df[df.kode.isin(["Z1", "Z2", "Z3"])]
    pilih = [
        ("B0", "B0"),
        ("A1", "A1 IndoBERT"),
        ("A2", "A2 CLIP"),
        ("F1", "F1 concat"),
        ("F2", "F2 gated"),
        ("F3", "F3 satu arah"),
        ("M1", "M1 In+CLIP"),
        ("M2", "M2 In+DINO"),
        ("T1", "T1 tuned"),
    ]
    rows, labels = [], []
    for kode, name in pilih:
        r = df[df.kode == kode]
        if kode == "A1":
            r = r[r.teks == "indobert"]
        if kode == "A2":
            r = r[r.gambar == "clip"]
        if len(r):
            rows.append(r.iloc[0])
            labels.append(name)
    if len(z):
        rows.append(
            pd.Series(
                {
                    "f1_true": z.f1_true.mean(),
                    "f1_ooc": z.f1_ooc.mean(),
                    "f1_mc": z.f1_mc.mean(),
                }
            )
        )
        labels.append("Final 3 seed")

    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, (col, name, color) in enumerate(
        [("f1_true", "true", BIRU), ("f1_ooc", "out-of-context", MERAH), ("f1_mc", "miscaptioned", HIJAU)]
    ):
        ax.bar(x + (i - 1) * w, [r[col] for r in rows], width=w, label=name, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.9)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("F1 per kelas di sepanjang tangga eksperimen")
    fig.tight_layout()
    fig.savefig(FIG / "14_per_kelas.png")
    plt.close(fig)


def fig_tuning(df: pd.DataFrame) -> None:
    t = df[df.kode.str.startswith("T") & df.kode.str[1:].str.isdigit()]
    if not len(t):
        return
    pivot = t.pivot_table(index="lr", columns="dropout", values="f1_macro", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"dropout {c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"lr {v:g}" for v in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > pivot.values.max() * 0.72 else "black")
    ax.set_title("Tuning: macro-F1 per learning rate dan dropout")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="macro-F1")
    fig.tight_layout()
    fig.savefig(FIG / "15_tuning_heatmap.png")
    plt.close(fig)


def fig_oof(df: pd.DataFrame) -> None:
    o = df[df.kode.isin([f"O{i}" for i in range(1, 6)])].sort_values("kode")
    if not len(o):
        return
    mean, std = o.f1_macro.mean(), o.f1_macro.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(o.kode, o.f1_macro, color=BIRU)
    ax.axhline(mean, color=MERAH, linestyle="--", linewidth=1.2, label=f"rata-rata {mean:.4f}")
    ax.axhspan(mean - std, mean + std, color=MERAH, alpha=0.08, label=f"±1 sd ({std:.4f})")
    for x, (_, r) in enumerate(o.iterrows()):
        ax.text(x, r.f1_macro + 0.006, f"{r.f1_macro:.3f}\nep {int(r.epoch_best)}",
                ha="center", fontsize=7)
    ax.set_ylim(0.6, 0.78)
    ax.set_ylabel("macro-F1")
    ax.set_title("Stabilitas 5-fold OOF")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "16_oof.png")
    plt.close(fig)


def fig_final(df: pd.DataFrame) -> None:
    z = df[df.kode.isin(["Z1", "Z2", "Z3"])].sort_values("kode")
    if not len(z):
        return
    mean, std = z.f1_macro.mean(), z.f1_macro.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(z.kode, z.f1_macro, color=HIJAU)
    ax.axhspan(mean - std, mean + std, color=MERAH, alpha=0.08, label=f"±1 sd ({std:.4f})")
    ax.axhline(mean, color=MERAH, linestyle="--", linewidth=1.2, label=f"rata-rata {mean:.4f}")
    for x, (_, r) in enumerate(z.iterrows()):
        ax.text(x, r.f1_macro + 0.004, f"{r.f1_macro:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.66, 0.76)
    ax.set_ylabel("macro-F1")
    ax.set_title("Tiga seed model final di test 700")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "17_final_seed.png")
    plt.close(fig)


def fig_cost(df: pd.DataFrame) -> None:
    d = df[(df.wall_clock_s.notna()) & (df.wall_clock_s > 0)].copy()
    if not len(d):
        return
    warna = {"F": BIRU, "M": MERAH, "T": HIJAU, "Z": KUNING}
    label_pilih = {"M1", "M2", "M3", "M4", "T1", "T2", "F1", "F2", "F3", "F5"}

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for prefix, color in warna.items():
        sub = d[d.kode.str.startswith(prefix)]
        ax.scatter(sub.wall_clock_s, sub.f1_macro, color=color, s=34,
                   label={"F": "fusion", "M": "grid", "T": "tuning", "Z": "final"}[prefix])
    for _, r in d.iterrows():
        if r.kode == "Z1":
            ax.annotate("Z1-3", (r.wall_clock_s, r.f1_macro), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        elif r.kode in label_pilih:
            ax.annotate(r.kode, (r.wall_clock_s, r.f1_macro), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("waktu latih (detik, termasuk early stopping)")
    ax.set_ylabel("macro-F1")
    ax.set_title("Waktu latih lawan performa")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "18_biaya.png")
    plt.close(fig)


def parse_epochs(path: Path) -> pd.DataFrame:
    nb = json.loads(path.read_text())
    rows = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = "".join(out.get("text", []))
            for m in re.finditer(r"(\S+) epoch (\d+) f1_macro ([0-9.]+)", text):
                rows.append((m.group(1), int(m.group(2)), float(m.group(3))))
    df = pd.DataFrame(rows, columns=["run_id", "epoch", "f1_macro"]).drop_duplicates()
    df["kode"] = df.run_id.str.split("_").str[0]
    return df


def fig_curves(df: pd.DataFrame) -> None:
    if not NB.exists():
        return
    cur = parse_epochs(NB)
    if not len(cur):
        return
    stages = [
        ("Fusion F0-F5", [f"F{i}" for i in range(6)]),
        ("Grid encoder M1-M4", [f"M{i}" for i in range(1, 5)]),
        ("Tuning T1-T6", [f"T{i}" for i in range(1, 7)]),
        ("OOF O1-O5", [f"O{i}" for i in range(1, 6)]),
    ]
    stages = [(t, k) for t, k in stages if len(cur[cur.kode.isin(k)])]
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), squeeze=False)
    for ax, (title, kodes) in zip(axes.ravel(), stages):
        sub = cur[cur.kode.isin(kodes)]
        for kode, g in sub.groupby("kode"):
            ax.plot(g.epoch, g.f1_macro, marker="o", markersize=3, linewidth=1.2, label=kode)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("F1 validasi")
        ax.legend(ncol=3, fontsize=6)
    fig.suptitle("Kurva validasi per epoch", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "19_kurva_validasi.png")
    plt.close(fig)


def parse_confusion(path: Path) -> np.ndarray | None:
    nb = json.loads(path.read_text())
    for cell in reversed(nb["cells"]):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            text = "".join(data.get("text/plain", []))
            lines = [l for l in text.splitlines() if l.strip().startswith("aktual:")]
            if len(lines) == 3:
                return np.array([[int(x) for x in re.findall(r"\d+", l)[:3]] for l in lines])
    return None


def fig_confusion(path: Path) -> None:
    cm = parse_confusion(path)
    if cm is None:
        return
    names = ["true", "out-of-context", "miscaptioned"]
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), names, rotation=20)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("prediksi")
    ax.set_ylabel("aktual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "black", fontsize=8)
    ax.set_title("Confusion matrix ensemble di test 700")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="proporsi per baris")
    fig.tight_layout()
    fig.savefig(FIG / "20_confusion_pct.png")
    plt.close(fig)


def main() -> int:
    df = load_ledger()
    print(f"ledger: {len(df)} run")
    fig_ladder(df)
    fig_per_class(df)
    fig_tuning(df)
    fig_oof(df)
    fig_final(df)
    fig_cost(df)
    fig_curves(df)
    fig_confusion(NB)
    out = sorted(FIG.glob("1[3-9]_*.png")) + sorted(FIG.glob("20_*.png"))
    for p in out:
        print("tulis:", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
