from __future__ import annotations

import collections
import difflib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "experiments" / "figures"
EXPORTS = ROOT / "data" / "audit_exports"
PAIRS = ROOT / "data" / "07_pairs.jsonl"
APP = Path(os.getenv("APP", Path.home() / "projects/college/ml/image-text-annotator"))

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.fontsize": 7,
    }
)

HIJAU = "#4a8c6f"
KUNING = "#c9a227"
MERAH = "#b5654a"
BIRU = "#3b6ea5"

URUT = ["cocok", "ragu", "tidak"]
WARNA = {"cocok": HIJAU, "ragu": KUNING, "tidak": MERAH}

TERLIHAT = {
    "bencana": ["banjir", "kebakaran", "tanah longsor", "longsor", "gempa bumi", "gempa",
                "kekeringan", "erupsi", "angin puting beliung"],
    "kendaraan": ["sepeda motor", "motor", "mobil", "truk", "bus", "kereta api", "kapal",
                  "sepeda", "pesawat"],
    "bangunan": ["rumah", "sekolah", "pasar", "kantor", "jembatan", "masjid", "gereja",
                 "gudang", "toko", "terminal", "stasiun", "bandara", "pelabuhan", "gedung",
                 "jalan", "lapangan", "taman", "tenda", "posko", "dermaga"],
    "orang": ["polisi", "tentara", "petugas", "relawan", "warga", "siswa", "guru",
              "pedagang", "nelayan", "petani", "petugas pemadam kebakaran"],
    "alam": ["sawah", "ladang", "hutan", "pantai", "sungai", "gunung", "danau", "perahu"],
}


def kategori(kata: str) -> str:
    for k, words in TERLIHAT.items():
        if kata in words:
            return k
    return "lain"


def load_exports() -> dict[str, dict]:
    files = sorted(EXPORTS.glob("anotasi-*.json"))
    best = {}
    for f in files:
        d = json.loads(f.read_text())
        nama = d["nama"]
        if nama not in best or len(d.get("test", {})) > len(best[nama].get("test", {})):
            best[nama] = d
    return best


def ac1(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    cnt = collections.Counter(x for p in pairs for x in p)
    m = 2 * n
    pi = {l: cnt[l] / m for l in URUT}
    pe = sum(pi[l] * (1 - pi[l]) for l in URUT) / (len(URUT) - 1)
    return (po - pe) / (1 - pe)


def fig_strata(ans: dict, meta: dict, po: float, ac: float) -> None:
    tab = collections.defaultdict(collections.Counter)
    for i, voters in ans.items():
        s = meta[i]["stratum"]
        for a in voters.values():
            tab[s][a] += 1
    strata = ["A", "B", "C"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    kiri = np.zeros(len(strata))
    for label in URUT:
        vals = np.array([tab[s][label] for s in strata])
        ax.barh(strata, vals, left=kiri, color=WARNA[label], label=label)
        for y, (v, l) in enumerate(zip(vals, kiri)):
            if v:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=8,
                        color="white" if label != "ragu" else "black")
        kiri += vals
    ax.invert_yaxis()
    ax.set_xlabel("jumlah rating")
    ax.set_title(f"Audit label true per strata sumber foto (kesepakatan {po:.0%}, AC1 {ac:.3f})")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(FIG / "21_audit_strata.png", bbox_inches="tight")
    plt.close(fig)


def fig_raters_per_rater(ans: dict, test_ans: dict, meta_test: dict) -> None:
    names = sorted({n for voters in ans.values() for n in voters})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    ax = axes[0]
    kiri = np.zeros(len(names))
    for label in URUT:
        vals = np.array([
            sum(1 for i in ans if n in ans[i] and ans[i][n] == label) for n in names
        ])
        ax.barh(names, vals, left=kiri, color=WARNA[label], label=label)
        kiri += vals
    ax.invert_yaxis()
    ax.set_title("Jawaban audit per penilai")
    ax.set_xlabel("jumlah item")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    ax = axes[1]
    rates, ns = [], []
    for n in names:
        mc = [i for i in test_ans if n in test_ans[i] and meta_test[i]["label"] == "miscaptioned"]
        rates.append(100 * sum(test_ans[i][n] == "tidak" for i in mc) / max(1, len(mc)))
        ns.append(len(mc))
    ax.bar(names, rates, color=BIRU)
    for x, (r, k) in enumerate(zip(rates, ns)):
        ax.text(x, r + 1.0, f"{r:.0f}%\nn={k}", ha="center", fontsize=7)
    ax.set_ylim(0, 45)
    ax.set_ylabel("MC dinilai tidak (%)")
    ax.set_title("Deteksi miscaptioned per penilai")
    fig.tight_layout()
    fig.savefig(FIG / "22_audit_raters.png")
    plt.close(fig)


def fig_test_classes(test_ans: dict, meta_test: dict) -> None:
    classes = ["true", "out-of-context", "miscaptioned"]
    tab = collections.defaultdict(collections.Counter)
    for i, voters in test_ans.items():
        lab = meta_test[i]["label"]
        for a in voters.values():
            tab[lab][a] += 1
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    kiri = np.zeros(len(classes))
    for label in URUT:
        vals = np.array([tab[c][label] for c in classes])
        ax.barh(classes, vals, left=kiri, color=WARNA[label], label=label)
        kiri += vals
    ax.invert_yaxis()
    ax.set_xlabel("jumlah rating")
    ax.set_title("Jawaban penilai pada set uji")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(FIG / "23_test_per_kelas.png")
    plt.close(fig)


def fig_mc_entity(test_ans: dict, meta_test: dict) -> None:
    true_caps = {}
    for line in PAIRS.open():
        r = json.loads(line)
        if r["label"] == "true":
            true_caps.setdefault(r["source"], r["caption"])

    tab = collections.defaultdict(collections.Counter)
    for i, voters in test_ans.items():
        item = meta_test[i]
        if item["label"] != "miscaptioned":
            continue
        asli = true_caps.get(item["source"], "")
        A, B = asli.split(), item["caption"].split()
        ops = [
            (" ".join(A[a1:a2]).lower(), " ".join(B[b1:b2]).lower())
            for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, A, B).get_opcodes()
            if tag == "replace"
        ]
        jenis = kategori(ops[0][0]) if len(ops) == 1 else "multi"
        for a in voters.values():
            tab[jenis][a] += 1

    urut = sorted(tab, key=lambda k: -sum(tab[k].values()))
    rates = [100 * tab[k]["tidak"] / sum(tab[k].values()) for k in urut]
    ns = [sum(tab[k].values()) for k in urut]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(urut, rates, color=MERAH)
    for x, (r, k) in enumerate(zip(rates, ns)):
        ax.text(x, r + 1.2, f"{r:.0f}%\nn={k}", ha="center", fontsize=7)
    ax.set_ylim(0, 50)
    ax.set_ylabel("MC dinilai tidak (%)")
    ax.set_title("Deteksi miscaptioned per jenis entitas yang diganti")
    fig.tight_layout()
    fig.savefig(FIG / "24_mc_entity.png")
    plt.close(fig)


def main() -> int:
    exports = load_exports()
    if len(exports) < 4:
        print(f"ekspor tidak lengkap: {len(exports)} dari 4")
        return 1

    items_audit = json.loads((APP / "private" / "items_audit.json").read_text())["items"]
    meta_audit = {it["id"]: it for it in items_audit}
    assignments = json.loads((APP / "private" / "assignments.json").read_text())
    meta_test = {it["id"]: it for it in assignments["test"]}

    ans = collections.defaultdict(dict)
    test_ans = collections.defaultdict(dict)
    for nama, d in exports.items():
        for i, a in d["audit"].items():
            ans[i][nama] = a
        for i, a in d["test"].items():
            test_ans[i][nama] = a

    pairs = [tuple(v.values()) for v in ans.values() if len(v) == 2]
    po = sum(a == b for a, b in pairs) / len(pairs)
    ac = ac1(pairs)
    print(f"audit: {len(ans)} item, {len(pairs)} ganda, kesepakatan {po:.3f}, AC1 {ac:.3f}")

    fig_strata(ans, meta_audit, po, ac)
    fig_raters_per_rater(ans, test_ans, meta_test)
    fig_test_classes(test_ans, meta_test)
    fig_mc_entity(test_ans, meta_test)

    for name in ["21_audit_strata.png", "22_audit_raters.png", "23_test_per_kelas.png", "24_mc_entity.png"]:
        print("tulis:", (FIG / name).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
