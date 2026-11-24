# %% [markdown]
# # Eksperimen Fusion Multimodal
#
# Notebook ini menjalankan seluruh skenario eksperimen Bab III. Setiap bab dibuka dengan
# tujuan, input, output, dan keputusan yang diambil, supaya alurnya terbaca sebelum kodenya.
#
# Cara menjalankan ada di `experiments/HANDOFF.md`. Ringkasnya:
#
# ```bash
# SMOKE=1 .venv-train/bin/python experiments/run_notebook.py
# .venv-train/bin/python experiments/run_notebook.py
# ```
#
# Tahap dijalankan lewat variabel lingkungan `RUN_BASELINE`, `RUN_GRID`, `RUN_FUSION`,
# `RUN_TUNING`, `RUN_OOF`, `RUN_FINAL`. Run yang sudah ada di ledger dilewati, kecuali
# `FORCE=1`.

# %% [markdown]
# ## 1. Persiapan
#
# **Tujuan** menyiapkan lingkungan, mengunci seed 727, dan mendeteksi lokasi repo.
#
# **Input** repo ini dan berkas `data/07_pairs.jsonl`.
#
# **Output** konfigurasi global, daftar versi library, dan device yang dipakai.
#
# **Keputusan** satu seed untuk seluruh eksplorasi, tiga seed hanya untuk model final.

# %%
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import replace
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path.cwd()
while not (ROOT / "data" / "07_pairs.jsonl").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.data import (  # noqa: E402
    LABEL_NAMES,
    PairDataset,
    clean_caption,
    load_pairs,
)
from experiments.lib.metrics import (  # noqa: E402
    append_ledger,
    count_params,
    done_ids,
    read_ledger,
    score,
)
from experiments.lib.run import Run, loader_generator, seed_worker, set_seed  # noqa: E402

SMOKE = os.getenv("SMOKE", "0") == "1"
RUN_BASELINE = os.getenv("RUN_BASELINE", "1") == "1"
RUN_GRID = os.getenv("RUN_GRID", "1") == "1"
RUN_FUSION = os.getenv("RUN_FUSION", "0") == "1"
RUN_TUNING = os.getenv("RUN_TUNING", "0") == "1"
RUN_OOF = os.getenv("RUN_OOF", "0") == "1"
RUN_FINAL = os.getenv("RUN_FINAL", "0") == "1"
FORCE = os.getenv("FORCE", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 727
MAX_LEN = 128
IMG_SIZE = 224

EXPERIMENTS = ROOT / "experiments"
FIG = EXPERIMENTS / "figures"
RUNS_DIR = EXPERIMENTS / ("runs_smoke" if SMOKE else "runs")
FIG.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = EXPERIMENTS / ("ledger_smoke.csv" if SMOKE else "ledger.csv")
PEMENANG = EXPERIMENTS / ("pemenang_smoke.json" if SMOKE else "pemenang.json")

set_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("device     :", DEVICE)
print("gpu        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torch      :", torch.__version__)
print("timm       :", timm.__version__)
print("ledger     :", LEDGER.relative_to(ROOT))
print("smoke      :", SMOKE)

# %% [markdown]
# ## 2. Dataset dan EDA
#
# **Tujuan** mengenali isi dataset sebelum modeling, tanpa mengubah apa pun.
#
# **Input** `data/07_pairs.jsonl`.
#
# **Output** tabel distribusi label, panjang caption, kategori, dan strata sumber foto.
#
# **Keputusan** tidak ada. Bab ini deskriptif, hasilnya dipakai untuk memilih `max_len`
# dan menjelaskan karakter data di Bab III.

# %%
rows = load_pairs()
df = pd.DataFrame(rows)
df["n_kata"] = df["caption"].str.split().str.len()

TAB = df["label"].value_counts().rename("jumlah").to_frame()
TAB["porsi"] = (TAB["jumlah"] / len(df)).round(4)
print("total pasangan:", len(df))
display(TAB)
display(df.groupby(["split", "label"]).size().unstack(fill_value=0))
display(df.groupby("fold")["label"].value_counts().unstack(fill_value=0))

# %%
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
order = ["true", "out-of-context", "miscaptioned"]
counts = df["label"].value_counts().reindex(order)
ax[0].bar(order, counts.values, color=["#3b6ea5", "#7a8b99", "#b5654a"])
ax[0].set_title("Distribusi label")
ax[0].tick_params(axis="x", rotation=20)

for label in order:
    ax[1].hist(df.loc[df.label == label, "n_kata"], bins=40, alpha=0.55, label=label)
ax[1].set_title("Panjang caption (kata)")
ax[1].legend(fontsize=7)

top = df["category"].value_counts().head(10)[::-1]
ax[2].barh(top.index, top.values, color="#3b6ea5")
ax[2].set_title("10 kategori terbesar")
fig.tight_layout()
fig.savefig(FIG / "02_distribusi.png", dpi=160)
plt.close(fig)

print(df.groupby("label")["n_kata"].describe()[["mean", "50%", "max"]].round(2))

# %%
def stratum(caption: str) -> str:
    import re

    c = caption or ""
    if re.match(r"\s*ilustrasi", c, re.I) or re.search(
        r"(shutterstock|pexels|unsplash|freepik|istock|getty|pixabay|canva|magnific"
        r"|ai generatif|dibuat dengan ai|generated by)",
        c,
        re.I,
    ):
        return "B"
    if re.search(r"(kompas|antara foto|antara/)", c, re.I):
        return "A"
    return "C"


df["stratum"] = df["caption"].map(stratum)
display(df.groupby(["label", "stratum"]).size().unstack(fill_value=0))

# %% [markdown]
# ## 3. Preprocessing
#
# **Tujuan** menyamakan bentuk input untuk semua encoder.
#
# **Input** caption mentah dan berkas gambar.
#
# **Output** tokenizer untuk IndoBERT dan mBERT, serta gambar 224 piksel dengan letterbox.
#
# **Keputusan** letterbox, bukan center crop, karena rasio asli 16:9 dan objek yang
# diperiksa sering berada di pinggir. Tidak ada augmentasi di versi ini, supaya semua
# model menerima input yang sama.

# %%
TOKENIZERS = {
    "indobert": AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1"),
    "mbert": AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
}
print({k: type(v).__name__ for k, v in TOKENIZERS.items()})

sample = df.iloc[0]
ids = TOKENIZERS["indobert"](clean_caption(sample["caption"]), truncation=True, max_length=MAX_LEN)
print("contoh caption :", clean_caption(sample["caption"])[:160])
print("jumlah token   :", len(ids["input_ids"]))

# %%
from PIL import Image
from experiments.lib.data import cache_path

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
with Image.open(cache_path(sample["image"])) as im:
    ax[0].imshow(im)
    ax[0].set_title(f"letterbox cache {im.size[0]}px")
    ax[0].axis("off")
ax[1].text(
    0.0,
    0.5,
    "caption:\n" + sample["caption"][:220] + "\n\n"
    "semua model menerima 224x224,\n"
    "bilah abu-abu di atas dan bawah",
    va="center",
    wrap=True,
    fontsize=9,
)
ax[1].axis("off")
fig.tight_layout()
fig.savefig(FIG / "03_contoh_input.png", dpi=160)
plt.close(fig)

# %% [markdown]
# ## 4. Pembagian data dan validasi
#
# **Tujuan** memastikan tidak ada kebocoran dan menetapkan peran tiap bagian data.
#
# **Input** kolom `split` dan `fold`.
#
# **Output** himpunan latih dan validasi untuk eksplorasi, serta test yang terkunci.
#
# **Keputusan** fold 0 menjadi validasi eksplorasi, karena baseline teks di Bab IV
# dihitung pada fold yang sama. Test 700 tidak dipakai sampai model final.

# %%
pool = df[df.split == "pool"].reset_index(drop=True)
test = df[df.split == "test"].reset_index(drop=True)
train_ex = pool[pool.fold != 0].reset_index(drop=True)
val_ex = pool[pool.fold == 0].reset_index(drop=True)

print("pool", len(pool), "| latih", len(train_ex), "| validasi", len(val_ex), "| test", len(test))

def irisan(a: pd.DataFrame, b: pd.DataFrame, col: str) -> int:
    return len(set(a[col]) & set(b[col]))

assert irisan(train_ex, val_ex, "image") == 0
assert irisan(train_ex, val_ex, "caption") == 0
assert irisan(pool, test, "image") == 0
assert irisan(pool, test, "caption") == 0
print("kebocoran gambar dan caption: 0")
print(val_ex.label.value_counts().to_dict())

# %% [markdown]
# ## 5. Model, loop latih, dan baseline
#
# **Tujuan** menetapkan lantai dan ablasi modalitas sebelum membandingkan kombinasi encoder.
#
# **Input** split dari bab 4.
#
# **Output** metrik B0, A1 (IndoBERT dan mBERT), dan A2 (CLIP dan DINOv2).
#
# **Keputusan** B0 dihitung tanpa latih. A2 diharapkan mendekati B0. Kalau A2 jauh di
# atasnya, ada kebocoran, bukan model bagus. Matriks embedding kata dibekukan untuk
# menghemat VRAM, sedangkan seluruh lapisan transformer tetap dilatih. Perlakuan ini sama
# untuk IndoBERT dan mBERT, jadi perbandingannya tetap setara.

# %%
TEXT_NAMES = {
    "indobert": "indobenchmark/indobert-base-p1",
    "mbert": "bert-base-multilingual-cased",
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


class TextEncoder(nn.Module):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_NAMES[key])
        self.dim = self.enc.config.hidden_size
        emb = getattr(self.enc, "embeddings", None)
        if emb is not None and hasattr(emb, "word_embeddings"):
            emb.word_embeddings.weight.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class ImageEncoder(nn.Module):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        if key == "clip":
            self.enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            self.dim = self.enc.config.hidden_size
            mean, std = CLIP_MEAN, CLIP_STD
        elif key == "dinov2":
            self.enc = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                img_size=IMG_SIZE,
                dynamic_img_size=True,
            )
            self.dim = self.enc.num_features
            mean, std = IN_MEAN, IN_STD
        else:
            raise ValueError(key)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, pixel):
        pixel = (pixel - self.mean) / self.std
        if self.key == "clip":
            return self.enc(pixel_values=pixel).last_hidden_state
        return self.enc.forward_features(pixel)


class TextOnly(nn.Module):
    def __init__(self, teks: str, dropout: float) -> None:
        super().__init__()
        self.text = TextEncoder(teks)
        d = 256
        self.head = nn.Sequential(
            nn.Linear(self.text.dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3)
        )

    def forward(self, batch):
        t = self.text(batch["input_ids"], batch["attention_mask"])
        return self.head(t[:, 0])


class ImageOnly(nn.Module):
    def __init__(self, gambar: str, dropout: float) -> None:
        super().__init__()
        self.image = ImageEncoder(gambar)
        d = 256
        self.head = nn.Sequential(
            nn.Linear(self.image.dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3)
        )

    def forward(self, batch):
        i = self.image(batch["pixel_values"])
        return self.head(i[:, 0])


class ConcatModel(nn.Module):
    def __init__(self, teks: str, gambar: str, dropout: float) -> None:
        super().__init__()
        self.text = TextEncoder(teks)
        self.image = ImageEncoder(gambar)
        d = 256
        self.pt = nn.Linear(self.text.dim, d)
        self.pi = nn.Linear(self.image.dim, d)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 3),
        )

    def forward(self, batch):
        t = self.text(batch["input_ids"], batch["attention_mask"])
        i = self.image(batch["pixel_values"])
        z = torch.cat([self.pt(t[:, 0]), self.pi(i[:, 0])], dim=-1)
        return self.head(z)


class CrossAttentionModel(nn.Module):
    """Cross-attention dua arah. uni hanya teks ke gambar."""

    def __init__(
        self,
        teks: str,
        gambar: str,
        dropout: float,
        heads: int = 4,
        layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.text = TextEncoder(teks)
        self.image = ImageEncoder(gambar)
        d = 256
        self.pt = nn.Linear(self.text.dim, d)
        self.pi = nn.Linear(self.image.dim, d)
        self.t2i = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
        )
        self.norm_t = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.i2t = (
            nn.ModuleList(
                [nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
            )
            if bidirectional
            else None
        )
        self.norm_i = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]) if bidirectional else None
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 3),
        )

    def forward(self, batch):
        t = self.pt(self.text(batch["input_ids"], batch["attention_mask"]))
        i = self.pi(self.image(batch["pixel_values"]))
        key_pad = ~batch["attention_mask"].bool()
        for k in range(len(self.t2i)):
            a, _ = self.t2i[k](t, i, i, need_weights=False)
            t = self.norm_t[k](t + a)
            if self.i2t is not None:
                b, _ = self.i2t[k](i, t, t, key_padding_mask=key_pad, need_weights=False)
                i = self.norm_i[k](i + b)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        t_pool = (t * mask).sum(1) / mask.sum(1).clamp(min=1)
        i_pool = i.mean(1)
        return self.head(torch.cat([t_pool, i_pool], dim=-1))


def build_model(run: Run) -> nn.Module:
    if run.teks and run.gambar and run.fusion == "concat":
        return ConcatModel(run.teks, run.gambar, run.dropout)
    if run.teks and run.gambar and run.fusion.startswith("xattn"):
        return CrossAttentionModel(
            run.teks,
            run.gambar,
            run.dropout,
            heads=8 if run.fusion == "xattn_bi_deep" else 4,
            layers=2 if run.fusion == "xattn_bi_deep" else 1,
            bidirectional=run.fusion != "xattn_uni",
        )
    if run.teks and not run.gambar:
        return TextOnly(run.teks, run.dropout)
    if run.gambar and not run.teks:
        return ImageOnly(run.gambar, run.dropout)
    raise ValueError(run)


def make_loader(subset, run: Run, shuffle: bool) -> DataLoader:
    tok = TOKENIZERS[run.teks or "indobert"]
    ds = PairDataset(subset, tok, max_len=MAX_LEN, image_size=IMG_SIZE)
    return DataLoader(
        ds,
        batch_size=run.batch,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=loader_generator(run.seed),
    )


def move(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys, ids = [], [], []
    for batch in loader:
        batch = move(batch)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(batch)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        ys.append(batch["label"].cpu().numpy())
        ids.append(batch["index"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(ids)


def class_weights(subset) -> torch.Tensor:
    counts = pd.Series([r["label"] for r in subset]).value_counts()
    w = torch.tensor([len(subset) / (3 * counts.get(name, 1)) for name in LABEL_NAMES], dtype=torch.float)
    return w.to(DEVICE)


def run_one(run: Run, train_rows, val_rows, epochs: int | None = None, save: bool = True) -> dict:
    if not FORCE and run.run_id in done_ids(LEDGER):
        print("lewati, sudah ada:", run.run_id)
        return next(r for r in read_ledger(LEDGER) if r["run_id"] == run.run_id)

    set_seed(run.seed)
    train_loader = make_loader(train_rows, run, shuffle=True)
    val_loader = make_loader(val_rows, run, shuffle=False)
    model = build_model(run).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=class_weights(train_rows))
    opt = torch.optim.AdamW(model.parameters(), lr=run.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    max_epochs = epochs or run.max_epochs
    best_f1, best_epoch, best_state, bad = -1.0, 0, None, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move(batch)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = crit(model(batch), batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        y, p, _ = predict(model, val_loader)
        f1 = f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        print(f"  {run.run_id} epoch {epoch} f1_macro {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= run.patience:
                break

    wall = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    y, p, ids = predict(model, val_loader)
    sc = score(y, p.argmax(1))

    if save:
        torch.save(best_state, RUNS_DIR / f"{run.run_id}.pt")
        np.savez(
            RUNS_DIR / f"{run.run_id}-val.npz",
            ids=ids,
            y=y,
            prob=p,
        )

    row = run.row() | {
        "epochs_run": epoch,
        "epoch_best": best_epoch,
        "acc": round(sc["acc"], 4),
        "bal_acc": round(sc["bal_acc"], 4),
        "f1_macro": round(sc["f1_macro"], 4),
        "f1_true": round(sc["f1_true"], 4),
        "f1_ooc": round(sc["f1_ooc"], 4),
        "f1_mc": round(sc["f1_mc"], 4),
        "val_n": len(val_rows),
        "wall_clock_s": round(wall, 1),
        "params": count_params(model),
        "vram_peak_mb": round(vram, 0),
        "catatan": "fold0-val",
    }
    append_ledger(row, LEDGER)
    del model, best_state, opt, crit, scaler, train_loader, val_loader
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return row

# %%
BASE = dict(lr=2e-5, dropout=0.1, batch=16, seed=SEED, max_epochs=5, patience=2)
RUNS_BASELINE = [
    Run("A1", "modalitas", "indobert", None, "none", **BASE),
    Run("A1", "modalitas", "mbert", None, "none", **BASE),
    Run("A2", "modalitas", None, "clip", "none", **BASE),
    Run("A2", "modalitas", None, "dinov2", "none", **BASE),
]

if SMOKE:
    train_rows = train_ex.groupby("label").head(100).to_dict("records")
    val_eval = val_ex.groupby("label").head(50)
    val_rows = val_eval.to_dict("records")
    RUNS_BASELINE = [replace(r, max_epochs=1, patience=1) for r in RUNS_BASELINE]
else:
    train_rows = train_ex.to_dict("records")
    val_rows = val_ex.to_dict("records")
    val_eval = val_ex

# B0 tanpa latih
mayor = pd.Series([r["label"] for r in train_rows]).value_counts().idxmax()
b0_pred = np.full(len(val_rows), LABEL_NAMES.index(mayor))
b0 = score(val_eval.label.map(LABEL_NAMES.index).to_numpy(), b0_pred)
b0_run = Run("B0", "modalitas", None, None, "none", **BASE)
if FORCE or b0_run.run_id not in done_ids(LEDGER):
    append_ledger(
        b0_run.row()
        | {
            "acc": round(b0["acc"], 4),
            "bal_acc": round(b0["bal_acc"], 4),
            "f1_macro": round(b0["f1_macro"], 4),
            "f1_true": round(b0["f1_true"], 4),
            "f1_ooc": round(b0["f1_ooc"], 4),
            "f1_mc": round(b0["f1_mc"], 4),
            "val_n": len(val_rows),
            "catatan": f"mayoritas={mayor}",
        },
        LEDGER,
    )
print("B0", {k: round(b0[k], 4) for k in ["acc", "f1_macro", "f1_mc"]})

if RUN_BASELINE:
    for run in RUNS_BASELINE:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_BASELINE mati")

# %% [markdown]
# ## 6. Grid encoder
#
# **Tujuan** membandingkan empat kombinasi encoder dengan fusion yang dikunci.
#
# **Input** split dari bab 4 dan konfigurasi dari bab 5.
#
# **Output** empat run M1 sampai M4, satu baris ledger per run.
#
# **Keputusan** fusion concat, hyperparameter sama untuk keempat, seed sama. Yang berubah
# hanya pasangan encoder.

# %%
RUNS_GRID = [
    Run("M1", "encoder", "indobert", "clip", "concat", **BASE),
    Run("M2", "encoder", "indobert", "dinov2", "concat", **BASE),
    Run("M3", "encoder", "mbert", "clip", "concat", **BASE),
    Run("M4", "encoder", "mbert", "dinov2", "concat", **BASE),
]
if SMOKE:
    RUNS_GRID = [replace(r, max_epochs=1, patience=1) for r in RUNS_GRID]

if RUN_GRID:
    for run in RUNS_GRID:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_GRID mati")

# %% [markdown]
# ## 7. Analisis tahap 1 dan penentuan pemenang
#
# **Tujuan** memilih kombinasi encoder terbaik secara jujur, lalu membekukannya untuk
# tahap 2.
#
# **Input** ledger berisi B0, A1, A2, dan M1 sampai M4.
#
# **Output** tabel perbandingan, grafik, dan `experiments/pemenang.json`.
#
# **Keputusan** pemenang diambil dari `f1_macro` tertinggi di antara M1 sampai M4.
# Kalau selisih dua teratas di bawah 0,02, dicatat bahwa tie-break seed diperlukan.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
for col in ["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc"]:
    led[col] = pd.to_numeric(led[col], errors="coerce")

tahap1 = led[led.kode.isin(["B0", "A1", "A2", "M1", "M2", "M3", "M4"])]
piv = tahap1.pivot_table(
    index="kode", values=["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc"]
).round(4)
display(piv)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
plot = tahap1.drop_duplicates("kode").sort_values("f1_macro")
ax.barh(plot["kode"], plot["f1_macro"], color="#3b6ea5")
ax.axvline(0.196, color="#b5654a", linestyle="--", linewidth=1)
ax.set_xlabel("macro-F1")
ax.set_title("Tahap 1: ablasi modalitas dan grid encoder")
fig.tight_layout()
fig.savefig(FIG / "06_grid.png", dpi=160)
plt.close(fig)

grid = led[led.kode.isin(["M1", "M2", "M3", "M4"])].drop_duplicates("kode")
grid = grid.sort_values("f1_macro", ascending=False).reset_index(drop=True)
if len(grid) == 0:
    print("belum ada hasil grid. Jalankan bab 6 dulu.")
else:
    gap = float(grid.loc[0, "f1_macro"] - grid.loc[1, "f1_macro"]) if len(grid) > 1 else 0.0
    kedua = grid.loc[1, "kode"] if len(grid) > 1 else "-"
    print("dua teratas:", grid.loc[0, "kode"], kedua, "| selisih", round(gap, 4))
    if gap < 0.02:
        print("selisih di bawah 0,02. Jalankan tie-break seed 728 dan 729 pada dua kandidat.")

    best = grid.loc[0]
    pemenang = {
        "kode": str(best["kode"]),
        "teks": str(best["teks"]),
        "gambar": str(best["gambar"]),
        "fusion": "concat",
        "lr": float(best["lr"]),
        "dropout": float(best["dropout"]),
        "batch": int(best["batch"]),
        "seed": int(best["seed"]),
        "epoch_best": int(best["epoch_best"]),
        "f1_macro": float(best["f1_macro"]),
        "selisih_dua_teratas": round(gap, 4),
        "ledger": LEDGER.name,
        "val_n": int(best["val_n"]),
    }
    PEMENANG.write_text(json.dumps(pemenang, indent=2))
    print(json.dumps(pemenang, indent=2))

# %% [markdown]
# ## 8. Perbandingan strategi fusion
#
# **Tujuan** menguji apakah interaksi antar modalitas menambah performa.
#
# **Input** `experiments/pemenang.json`.
#
# **Output** empat run F0 sampai F3 pada pasangan encoder pemenang.
#
# **Keputusan** satu variabel berubah, yaitu fusion. Encoder, hyperparameter, dan seed
# dikunci.

# %%
P = json.loads(PEMENANG.read_text())
RUNS_FUSION = [
    Run("F0", "fusion", P["teks"], P["gambar"], "concat", lr=P["lr"], dropout=P["dropout"], batch=P["batch"], seed=SEED),
    Run("F1", "fusion", P["teks"], P["gambar"], "xattn_uni", lr=P["lr"], dropout=P["dropout"], batch=P["batch"], seed=SEED),
    Run("F2", "fusion", P["teks"], P["gambar"], "xattn_bi", lr=P["lr"], dropout=P["dropout"], batch=P["batch"], seed=SEED),
    Run("F3", "fusion", P["teks"], P["gambar"], "xattn_bi_deep", lr=P["lr"], dropout=P["dropout"], batch=P["batch"], seed=SEED),
]
if SMOKE:
    RUNS_FUSION = [replace(r, max_epochs=1, patience=1) for r in RUNS_FUSION]

if RUN_FUSION:
    for run in RUNS_FUSION:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_FUSION mati")

# %% [markdown]
# ## 9. Faktor hyperparameter
#
# **Tujuan** mencari learning rate dan dropout terbaik pada konfigurasi fusion pemenang.
#
# **Input** hasil bab 8.
#
# **Output** enam run T1 sampai T6.
#
# **Keputusan** grid dua parameter, bukan OFAT, karena keduanya berinteraksi. Batch dan
# jumlah epoch dikunci supaya waktu tetap terkendali.

# %%
fus = pd.DataFrame(read_ledger(LEDGER))
fus["f1_macro"] = pd.to_numeric(fus["f1_macro"], errors="coerce")
kandidat = fus[fus.kode.isin(["F0", "F1", "F2", "F3"])]
if len(kandidat):
    best_fusion = kandidat.sort_values("f1_macro", ascending=False).iloc[0]
    FUSION_KEY = str(best_fusion["fusion"])
    print("fusion terbaik:", best_fusion["kode"], FUSION_KEY)
else:
    FUSION_KEY = "concat"
    print("belum ada hasil fusion, pakai concat")

RUNS_TUNING = []
i = 1
for lr in [1e-5, 2e-5, 5e-5]:
    for do in [0.1, 0.3]:
        RUNS_TUNING.append(
            Run(
                f"T{i}", "tuning", P["teks"], P["gambar"], FUSION_KEY,
                lr=lr, dropout=do, batch=P["batch"], seed=SEED,
            )
        )
        i += 1
if SMOKE:
    RUNS_TUNING = [replace(r, max_epochs=1, patience=1) for r in RUNS_TUNING[:2]]

if RUN_TUNING:
    for run in RUNS_TUNING:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_TUNING mati")

# %% [markdown]
# ## 10. Final 5-fold OOF
#
# **Tujuan** mengukur stabilitas antar fold dan menyiapkan prediksi out-of-fold untuk
# metrik pairwise.
#
# **Input** konfigurasi final dari bab 9.
#
# **Output** lima run O1 sampai O5, satu berkas `oof.npz`, dan tabel per fold.
#
# **Keputusan** satu seed. Early stopping tetap dipakai di sini, dan median `epoch_best`
# menjadi jumlah epoch untuk model final.

# %%
tun = pd.DataFrame(read_ledger(LEDGER))
tun["f1_macro"] = pd.to_numeric(tun["f1_macro"], errors="coerce")
kandidat_t = tun[tun.kode.str.startswith("T")]
if len(kandidat_t):
    final_cfg = kandidat_t.sort_values("f1_macro", ascending=False).iloc[0]
    FINAL_LR = float(final_cfg["lr"])
    FINAL_DO = float(final_cfg["dropout"])
    FINAL_FUSION = str(final_cfg["fusion"])
    print("konfigurasi final:", final_cfg["kode"], FINAL_LR, FINAL_DO, FINAL_FUSION)
else:
    FINAL_LR, FINAL_DO, FINAL_FUSION = float(P["lr"]), float(P["dropout"]), FUSION_KEY
    print("belum ada tuning, pakai konfigurasi pemenang:", FINAL_LR, FINAL_DO, FINAL_FUSION)

OOF_IDS = []
for k in range(5):
    OOF_IDS.append(
        Run(
            f"O{k + 1}", "oof", P["teks"], P["gambar"], FINAL_FUSION,
            lr=FINAL_LR, dropout=FINAL_DO,
            batch=int(P["batch"]), seed=SEED,
        )
    )
if SMOKE:
    OOF_IDS = [replace(r, max_epochs=1, patience=1) for r in OOF_IDS]

if RUN_OOF:
    for run, k in zip(OOF_IDS, range(5)):
        tr_df = pool[pool.fold != k].reset_index(drop=True)
        va_df = pool[pool.fold == k].reset_index(drop=True)
        if SMOKE:
            tr_df = tr_df.groupby("label").head(100)
            va_df = va_df.groupby("label").head(50)
        run_one(run, tr_df.to_dict("records"), va_df.to_dict("records"))
    parts = [np.load(RUNS_DIR / f"{r.run_id}-val.npz") for r in OOF_IDS]
    np.savez(
        RUNS_DIR / "oof.npz",
        ids=np.concatenate([p["ids"] for p in parts]),
        y=np.concatenate([p["y"] for p in parts]),
        prob=np.concatenate([p["prob"] for p in parts]),
    )
    print("oof.npz tersimpan")
else:
    print("RUN_OOF mati")

# %% [markdown]
# ## 11. Model final dan uji 700
#
# **Tujuan** menghasilkan angka utama yang masuk kesimpulan.
#
# **Input** seluruh pool, test 700, dan jumlah epoch dari median OOF.
#
# **Output** tiga run Z1 sampai Z3, prediksi test, rata-rata dan simpangan.
#
# **Keputusan** tiga seed 727, 728, 729. Tidak ada validasi di sini, karena peran itu
# sudah selesai di bab 10. Test pertama kali dibuka di bab ini.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
led["epoch_best"] = pd.to_numeric(led["epoch_best"], errors="coerce")
oof_rows = led[led.kode.str.startswith("O")]
EPOCHS_FINAL = int(median(oof_rows["epoch_best"].tolist())) if len(oof_rows) else int(P["epoch_best"])
print("epoch final:", EPOCHS_FINAL)

FINAL_SEEDS = [SEED, SEED + 1, SEED + 2]
FINAL = [
    Run(f"Z{i + 1}", "final", P["teks"], P["gambar"], FINAL_FUSION,
        lr=FINAL_LR, dropout=FINAL_DO,
        batch=int(P["batch"]), seed=s)
    for i, s in enumerate(FINAL_SEEDS)
]

if RUN_FINAL:
    pool_final = pool.head(300).to_dict("records") if SMOKE else pool.to_dict("records")
    test_eval = test.head(100) if SMOKE else test
    test_final = test_eval.to_dict("records")
    epochs_final = 1 if SMOKE else EPOCHS_FINAL
    for run in FINAL:
        if not FORCE and run.run_id in done_ids(LEDGER):
            print("lewati, sudah ada:", run.run_id)
            continue
        set_seed(run.seed)
        tr_loader = make_loader(pool_final, run, shuffle=True)
        te_loader = make_loader(test_final, run, shuffle=False)
        model = build_model(run).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=class_weights(pool_final))
        opt = torch.optim.AdamW(model.parameters(), lr=run.lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for epoch in range(1, epochs_final + 1):
            model.train()
            for batch in tr_loader:
                batch = move(batch)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    loss = crit(model(batch), batch["label"])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
        y, p, ids = predict(model, te_loader)
        np.savez(RUNS_DIR / f"{run.run_id}-test.npz", ids=ids, y=y, prob=p)
        sc = score(y, p.argmax(1))
        append_ledger(
            run.row()
            | {
                "epochs_run": epochs_final,
                "epoch_best": epochs_final,
                "acc": round(sc["acc"], 4),
                "bal_acc": round(sc["bal_acc"], 4),
                "f1_macro": round(sc["f1_macro"], 4),
                "f1_true": round(sc["f1_true"], 4),
                "f1_ooc": round(sc["f1_ooc"], 4),
                "f1_mc": round(sc["f1_mc"], 4),
                "val_n": len(test_final),
                "wall_clock_s": round(time.perf_counter() - t0, 1),
                "params": count_params(model),
                "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 0),
                "catatan": "test-700",
            },
            LEDGER,
        )
        del model
        torch.cuda.empty_cache()
else:
    print("RUN_FINAL mati")

# %% [markdown]
# ## 12. Analisis error dan ringkasan
#
# **Tujuan** menjelaskan jenis kesalahan dan batasannya.
#
# **Input** prediksi test dari tiga seed dan ledger.
#
# **Output** confusion matrix, contoh kesalahan, dan tabel ringkasan.
#
# **Keputusan** kelas MC dilaporkan bersama subset yang terverifikasi manusia, karena
# audit menunjukkan sebagian item MC tidak bisa diperiksa dari gambar.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
for col in ["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc"]:
    led[col] = pd.to_numeric(led[col], errors="coerce")

ringkas = led.drop_duplicates("kode")[["kode", "faktor", "teks", "gambar", "fusion", "f1_macro", "acc", "bal_acc", "params"]]
display(ringkas.sort_values(["faktor", "kode"]))

test_files = sorted(RUNS_DIR.glob("Z*-test.npz"))
if test_files:
    test_eval = test.head(100) if SMOKE else test
    probs = np.mean([np.load(f)["prob"] for f in test_files], axis=0)
    y = np.load(test_files[0])["y"]
    pred = probs.argmax(1)
    sc = score(y, pred)
    print({k: round(v, 4) for k, v in sc.items() if k != "confusion"})
    print(pd.DataFrame(sc["confusion"], index=LABEL_NAMES, columns=LABEL_NAMES))

    err = test_eval.assign(pred=[LABEL_NAMES[i] for i in pred], conf=probs.max(1))
    err = err[err.label != err.pred].sort_values("conf", ascending=False)
    display(err[["label", "pred", "conf", "caption"]].head(12))
else:
    print("belum ada prediksi test, jalankan RUN_FINAL=1")
