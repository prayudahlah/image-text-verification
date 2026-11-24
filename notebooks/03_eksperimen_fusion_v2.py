# %% [markdown]
# # Eksperimen Fusion Multimodal, versi 2
#
# Versi ini menggantikan v1. Perbedaannya tiga.
#
# 1. Protokol latih memakai warmup 10 persen dan cosine decay, batas 8 epoch, patience 3.
# 2. Urutan dibalik. Fusion diuji lebih dulu pada satu kombinasi acuan, lalu grid encoder
#    memakai metode fusion yang menang.
# 3. Metrik pairwise true lawan MC dilaporkan di samping macro-F1.
#
# Versi 1 tetap ada di `notebooks/02_eksperimen_fusion_v1.ipynb` sebagai pilot dan bukti
# pemilihan protokol.
#
# Cara menjalankan:
#
# ```bash
# SMOKE=1 .venv-train/bin/python experiments/run_notebook.py
# FUSION_ONLY=F1 .venv-train/bin/python experiments/run_notebook.py
# .venv-train/bin/python experiments/run_notebook.py
# ```

# %% [markdown]
# ## 1. Persiapan
#
# **Tujuan** menyiapkan lingkungan, mengunci seed 727, dan menetapkan protokol latih.
#
# **Input** repo ini, `data/07_pairs.jsonl`, dan `experiments/v1/pemenang.json`.
#
# **Output** konfigurasi global, daftar versi library, dan device.
#
# **Keputusan** warmup 10 persen dan cosine decay, 8 epoch, patience 3. Protokol ini
# dipilih setelah pilot v1 menunjukkan kurva validasi yang naik-turun tanpa scheduler.

# %%
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

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
RUN_FUSION = os.getenv("RUN_FUSION", "1") == "1"
RUN_GRID = os.getenv("RUN_GRID", "0") == "1"
RUN_TUNING = os.getenv("RUN_TUNING", "0") == "1"
RUN_OOF = os.getenv("RUN_OOF", "0") == "1"
RUN_FINAL = os.getenv("RUN_FINAL", "0") == "1"
FUSION_ONLY = os.getenv("FUSION_ONLY", "").strip()
FORCE = os.getenv("FORCE", "0") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 727
MAX_LEN = 128
IMG_SIZE = 224
WARMUP = 0.1

EXPERIMENTS = ROOT / "experiments"
FIG = EXPERIMENTS / "figures"
RUNS_DIR = EXPERIMENTS / ("runs_smoke" if SMOKE else "runs")
FIG.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = EXPERIMENTS / ("ledger_smoke.csv" if SMOKE else "ledger.csv")
PEMENANG_FUSION = EXPERIMENTS / ("pemenang_fusion_smoke.json" if SMOKE else "pemenang_fusion.json")
PEMENANG = EXPERIMENTS / ("pemenang_smoke.json" if SMOKE else "pemenang.json")

set_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("device     :", DEVICE)
print("gpu        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("torch      :", torch.__version__)
print("ledger     :", LEDGER.relative_to(ROOT))
print("smoke      :", SMOKE)
print("fusion only:", FUSION_ONLY or "-")

# %% [markdown]
# ## 2. Dataset dan EDA
#
# **Tujuan** mengenali isi dataset sebelum modeling.
#
# **Input** `data/07_pairs.jsonl`.
#
# **Output** tabel distribusi label, panjang caption, kategori, dan strata.
#
# **Keputusan** tidak ada. Bab ini deskriptif.

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
# **Output** tokenizer IndoBERT dan mBERT, serta gambar 224 piksel dengan letterbox.
#
# **Keputusan** letterbox, bukan center crop, karena rasio asli 16:9. Tanpa augmentasi.

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
# **Output** latih dan validasi eksplorasi, serta test yang terkunci.
#
# **Keputusan** fold 0 menjadi validasi eksplorasi. Test 700 tidak dipakai sampai model
# final.

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
# **Tujuan** menetapkan lantai dan ablasi modalitas sebelum membandingkan strategi fusion.
#
# **Input** split dari bab 4.
#
# **Output** metrik B0, A1 (IndoBERT, mBERT), dan A2 (CLIP, DINOv2).
#
# **Keputusan** B0 tanpa latih. Matriks embedding kata dibekukan untuk menghemat VRAM,
# lapisan transformer tetap dilatih, dan perlakuan ini sama untuk semua run teks.

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


class LateFusionModel(nn.Module):
    def __init__(self, teks: str, gambar: str, dropout: float) -> None:
        super().__init__()
        self.text = TextEncoder(teks)
        self.image = ImageEncoder(gambar)
        d = 256
        self.ht = nn.Sequential(
            nn.Linear(self.text.dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3)
        )
        self.hi = nn.Sequential(
            nn.Linear(self.image.dim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3)
        )

    def forward(self, batch):
        t = self.text(batch["input_ids"], batch["attention_mask"])
        i = self.image(batch["pixel_values"])
        return 0.5 * (self.ht(t[:, 0]) + self.hi(i[:, 0]))


class GatedModel(nn.Module):
    def __init__(self, teks: str, gambar: str, dropout: float) -> None:
        super().__init__()
        self.text = TextEncoder(teks)
        self.image = ImageEncoder(gambar)
        d = 256
        self.pt = nn.Linear(self.text.dim, d)
        self.pi = nn.Linear(self.image.dim, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3)
        )

    def forward(self, batch):
        t = self.pt(self.text(batch["input_ids"], batch["attention_mask"])[:, 0])
        i = self.pi(self.image(batch["pixel_values"])[:, 0])
        g = self.gate(torch.cat([t, i], dim=-1))
        return self.head(g * t + (1 - g) * i)


class CrossAttentionModel(nn.Module):
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
    if run.teks and not run.gambar:
        return TextOnly(run.teks, run.dropout)
    if run.gambar and not run.teks:
        return ImageOnly(run.gambar, run.dropout)
    if run.fusion == "late":
        return LateFusionModel(run.teks, run.gambar, run.dropout)
    if run.fusion == "concat":
        return ConcatModel(run.teks, run.gambar, run.dropout)
    if run.fusion == "gated":
        return GatedModel(run.teks, run.gambar, run.dropout)
    if run.fusion.startswith("xattn"):
        return CrossAttentionModel(
            run.teks,
            run.gambar,
            run.dropout,
            heads=8 if run.fusion == "xattn_bi_deep" else 4,
            layers=2 if run.fusion == "xattn_bi_deep" else 1,
            bidirectional=run.fusion != "xattn_uni",
        )
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


def build_pairs(subset) -> list[tuple[int, int]]:
    idx = defaultdict(list)
    for i, r in enumerate(subset):
        idx[(r["source"], r["image"])].append(i)
    pairs = []
    for _, ids in idx.items():
        ts = [i for i in ids if subset[i]["label"] == "true"]
        ms = [i for i in ids if subset[i]["label"] == "miscaptioned"]
        if ts and ms:
            for m in ms:
                pairs.append((ts[0], m))
    return pairs


def pairwise_acc(run_id: str, pairs: list[tuple[int, int]]) -> float:
    path = RUNS_DIR / f"{run_id}-val.npz"
    if not pairs or not path.exists():
        return float("nan")
    p = np.load(path)["prob"][:, 0]
    return float(np.mean([p[a] > p[b] for a, b in pairs]))


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
    steps = len(train_loader) * max_epochs
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=int(WARMUP * steps), num_training_steps=steps
    )

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
            sched.step()
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
        np.savez(RUNS_DIR / f"{run.run_id}-val.npz", ids=ids, y=y, prob=p)

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
    del model, best_state, opt, crit, scaler, train_loader, val_loader, sched
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return row

# %%
BASE = dict(lr=2e-5, dropout=0.1, batch=16, seed=SEED, max_epochs=8, patience=3)
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

PAIRS = build_pairs(val_rows)

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
# ## 6. Perbandingan strategi fusion
#
# **Tujuan** menguji apakah interaksi antar modalitas menambah performa, dan pada level
# mana.
#
# **Input** kombinasi acuan dari pilot v1, yaitu M4 mBERT dan DINOv2.
#
# **Output** enam run F0 sampai F5.
#
# **Keputusan** kombinasi encoder dikunci, yang berubah hanya fusion. Urutannya naik dari
# keputusan, ke fitur global, ke fitur berbobot, ke token.

# %%
REF_PATH = EXPERIMENTS / "v1" / "pemenang.json"
if REF_PATH.exists():
    REF = json.loads(REF_PATH.read_text())
else:
    REF = {"kode": "M4", "teks": "mbert", "gambar": "dinov2"}
print("kombinasi acuan:", REF["kode"], REF["teks"], REF["gambar"])

FUSIONS = [
    ("F0", "late"),
    ("F1", "concat"),
    ("F2", "gated"),
    ("F3", "xattn_uni"),
    ("F4", "xattn_bi"),
    ("F5", "xattn_bi_deep"),
]
RUNS_FUSION = [
    Run(kode, "fusion", REF["teks"], REF["gambar"], fusion, **BASE) for kode, fusion in FUSIONS
]
if FUSION_ONLY:
    RUNS_FUSION = [r for r in RUNS_FUSION if r.kode == FUSION_ONLY]
if SMOKE:
    RUNS_FUSION = [replace(r, max_epochs=1, patience=1) for r in RUNS_FUSION]

if RUN_FUSION:
    for run in RUNS_FUSION:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_FUSION mati")

# %% [markdown]
# ## 7. Analisis fusion dan keputusan
#
# **Tujuan** memilih metode fusion yang dipakai di grid encoder.
#
# **Input** ledger berisi F0 sampai F5.
#
# **Output** tabel, grafik, dan `experiments/pemenang_fusion.json`.
#
# **Keputusan** pemenang diambil dari `f1_macro` tertinggi. Kalau selisih dua teratas di
# bawah 0,02, keduanya dianggap seri dan pemenang ditentukan oleh kesesuaian dengan
# pertanyaan penelitian, yaitu cross-attention satu arah, selama skornya masih di dalam
# 0,02 dari yang tertinggi. Aturan ini dinyatakan sebelum grid dijalankan, supaya
# pemilihan tidak bergantung pada hasil akhir.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
for col in ["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc"]:
    led[col] = pd.to_numeric(led[col], errors="coerce")

fus = led[led.kode.isin([f"F{i}" for i in range(6)])].drop_duplicates("kode").copy()
fus["pairwise"] = [pairwise_acc(r, PAIRS) for r in fus["run_id"]]
display(fus[["kode", "fusion", "acc", "bal_acc", "f1_macro", "f1_mc", "pairwise", "epoch_best"]])

if len(fus):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    d = fus.sort_values("f1_macro")
    ax[0].barh(d["kode"] + " " + d["fusion"], d["f1_macro"], color="#3b6ea5")
    ax[0].set_title("macro-F1 per strategi fusion")
    ax[1].barh(d["kode"] + " " + d["fusion"], d["pairwise"], color="#b5654a")
    ax[1].set_xlim(0.5, 1.0)
    ax[1].set_title("pairwise true lawan MC")
    fig.tight_layout()
    fig.savefig(FIG / "07_fusion.png", dpi=160)
    plt.close(fig)

    ranked = fus.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    gap = float(ranked.loc[0, "f1_macro"] - ranked.loc[1, "f1_macro"]) if len(ranked) > 1 else 0.0
    best = ranked.loc[0]
    aturan = "macro-F1 tertinggi"

    # Tie-break yang dinyatakan sebelum grid: kalau selisih dua teratas di bawah 0,02,
    # pilih strategi yang menjadi fokus penelitian selama skornya masih di dalam 0,02.
    STUDI = "xattn_uni"
    if gap < 0.02:
        kandidat = fus[fus.fusion == STUDI]
        if len(kandidat):
            k = kandidat.iloc[0]
            if float(best["f1_macro"]) - float(k["f1_macro"]) < 0.02:
                best = k
                aturan = "seri, dipilih strategi penelitian"

    pemenang_fusion = {
        "kode": str(best["kode"]),
        "fusion": str(best["fusion"]),
        "f1_macro": float(best["f1_macro"]),
        "pairwise": float(best["pairwise"]) if pd.notna(best["pairwise"]) else None,
        "epoch_best": int(best["epoch_best"]),
        "selisih_dua_teratas": round(gap, 4),
        "aturan": aturan,
        "acuan_encoder": {"teks": REF["teks"], "gambar": REF["gambar"]},
        "ledger": LEDGER.name,
    }
    PEMENANG_FUSION.write_text(json.dumps(pemenang_fusion, indent=2))
    print(json.dumps(pemenang_fusion, indent=2))
else:
    print("belum ada hasil fusion")

# %% [markdown]
# ## 8. Grid encoder dengan fusion pemenang
#
# **Tujuan** membandingkan empat kombinasi encoder memakai metode fusion yang menang.
#
# **Input** `experiments/pemenang_fusion.json`.
#
# **Output** empat run M1 sampai M4, tabel, grafik, dan `experiments/pemenang.json`.
#
# **Keputusan** fusion dikunci ke pemenang bab 7. Hyperparameter dan seed sama untuk
# keempat. Yang berubah hanya pasangan encoder.

# %%
P_FUS = json.loads(PEMENANG_FUSION.read_text()) if PEMENANG_FUSION.exists() else {"fusion": "concat"}
FUS = P_FUS["fusion"]
print("fusion untuk grid:", FUS)

RUNS_GRID = [
    Run("M1", "encoder", "indobert", "clip", FUS, **BASE),
    Run("M2", "encoder", "indobert", "dinov2", FUS, **BASE),
    Run("M3", "encoder", "mbert", "clip", FUS, **BASE),
    Run("M4", "encoder", "mbert", "dinov2", FUS, **BASE),
]
if SMOKE:
    RUNS_GRID = [replace(r, max_epochs=1, patience=1) for r in RUNS_GRID]

if RUN_GRID:
    for run in RUNS_GRID:
        run_one(run, train_rows, val_rows)
else:
    print("RUN_GRID mati")

# %%
led = pd.DataFrame(read_ledger(LEDGER))
grid = led[led.kode.isin(["M1", "M2", "M3", "M4"])].drop_duplicates("kode").copy()
if len(grid):
    grid["pairwise"] = [pairwise_acc(r, PAIRS) for r in grid["run_id"]]
    display(grid[["kode", "teks", "gambar", "fusion", "acc", "bal_acc", "f1_macro", "f1_mc", "pairwise", "epoch_best"]])
    ranked = grid.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    gap = float(ranked.loc[0, "f1_macro"] - ranked.loc[1, "f1_macro"]) if len(ranked) > 1 else 0.0
    best = ranked.loc[0]
    pemenang = {
        "kode": str(best["kode"]),
        "teks": str(best["teks"]),
        "gambar": str(best["gambar"]),
        "fusion": str(best["fusion"]),
        "lr": float(best["lr"]),
        "dropout": float(best["dropout"]),
        "batch": int(best["batch"]),
        "seed": int(best["seed"]),
        "epoch_best": int(best["epoch_best"]),
        "f1_macro": float(best["f1_macro"]),
        "pairwise": float(best["pairwise"]) if pd.notna(best["pairwise"]) else None,
        "selisih_dua_teratas": round(gap, 4),
    }
    PEMENANG.write_text(json.dumps(pemenang, indent=2))
    print(json.dumps(pemenang, indent=2))
else:
    print("belum ada hasil grid")

# %% [markdown]
# ## 9. Faktor hyperparameter
#
# **Tujuan** mencari learning rate dan dropout terbaik pada konfigurasi final.
#
# **Input** pemenang encoder dan fusion.
#
# **Output** enam run T1 sampai T6.
#
# **Keputusan** grid dua parameter, bukan OFAT.

# %%
tun = pd.DataFrame(read_ledger(LEDGER))
tun["f1_macro"] = pd.to_numeric(tun["f1_macro"], errors="coerce")
P = json.loads(PEMENANG.read_text()) if PEMENANG.exists() else {
    "teks": REF["teks"], "gambar": REF["gambar"], "fusion": FUS,
    "lr": BASE["lr"], "dropout": BASE["dropout"], "batch": BASE["batch"],
    "epoch_best": 4,
}

RUNS_TUNING = []
i = 1
for lr in [1e-5, 2e-5, 5e-5]:
    for do in [0.1, 0.3]:
        RUNS_TUNING.append(
            Run(f"T{i}", "tuning", P["teks"], P["gambar"], P["fusion"],
                lr=lr, dropout=do, batch=P["batch"], seed=SEED)
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
# **Tujuan** mengukur stabilitas dan menyiapkan prediksi out-of-fold untuk pairwise.
#
# **Input** konfigurasi final.
#
# **Output** lima run O1 sampai O5, `oof.npz`, dan tabel per fold.
#
# **Keputusan** satu seed, early stopping tetap dipakai, median `epoch_best` jadi epoch
# model final.

# %%
tun = pd.DataFrame(read_ledger(LEDGER))
tun["f1_macro"] = pd.to_numeric(tun["f1_macro"], errors="coerce")
kandidat_t = tun[tun.kode.str.startswith("T")]
if len(kandidat_t):
    final_cfg = kandidat_t.sort_values("f1_macro", ascending=False).iloc[0]
    FINAL_LR, FINAL_DO, FINAL_FUSION = float(final_cfg["lr"]), float(final_cfg["dropout"]), str(final_cfg["fusion"])
    print("konfigurasi final:", final_cfg["kode"], FINAL_LR, FINAL_DO, FINAL_FUSION)
else:
    FINAL_LR, FINAL_DO, FINAL_FUSION = float(P["lr"]), float(P["dropout"]), str(P["fusion"])
    print("belum ada tuning, pakai konfigurasi pemenang:", FINAL_LR, FINAL_DO, FINAL_FUSION)

OOF_IDS = [
    Run(f"O{k + 1}", "oof", P["teks"], P["gambar"], FINAL_FUSION,
        lr=FINAL_LR, dropout=FINAL_DO, batch=int(P["batch"]), seed=SEED)
    for k in range(5)
]
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
# **Tujuan** menghasilkan angka utama.
#
# **Input** seluruh pool, test 700, dan epoch dari median OOF.
#
# **Output** tiga run Z1 sampai Z3, prediksi test, rata-rata dan simpangan.
#
# **Keputusan** seed 727, 728, 729, tanpa validasi, epoch tetap.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
led["epoch_best"] = pd.to_numeric(led["epoch_best"], errors="coerce")
oof_rows = led[led.kode.str.startswith("O")]
EPOCHS_FINAL = int(median(oof_rows["epoch_best"].tolist())) if len(oof_rows) else int(P.get("epoch_best", 4))
print("epoch final:", EPOCHS_FINAL)

FINAL_SEEDS = [SEED, SEED + 1, SEED + 2]
FINAL = [
    Run(f"Z{i + 1}", "final", P["teks"], P["gambar"], FINAL_FUSION,
        lr=FINAL_LR, dropout=FINAL_DO, batch=int(P["batch"]), seed=s)
    for i, s in enumerate(FINAL_SEEDS)
]

if RUN_FINAL:
    pool_final = pool.groupby("label").head(100).to_dict("records") if SMOKE else pool.to_dict("records")
    test_eval = test.groupby("label").head(35) if SMOKE else test
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
        steps = len(tr_loader) * epochs_final
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=int(WARMUP * steps), num_training_steps=steps
        )
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
                sched.step()
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
        del model, opt, crit, scaler, tr_loader, te_loader, sched
        torch.cuda.empty_cache()
else:
    print("RUN_FINAL mati")

# %% [markdown]
# ## 12. Analisis error dan ringkasan
#
# **Tujuan** menjelaskan jenis kesalahan dan batasannya.
#
# **Input** prediksi test tiga seed dan ledger.
#
# **Output** confusion matrix, contoh kesalahan, dan ringkasan.
#
# **Keputusan** kelas MC dilaporkan bersama batasan hasil audit, karena sebagian item MC
# tidak bisa diperiksa dari gambar.

# %%
led = pd.DataFrame(read_ledger(LEDGER))
for col in ["acc", "bal_acc", "f1_macro", "f1_true", "f1_ooc", "f1_mc"]:
    led[col] = pd.to_numeric(led[col], errors="coerce")

ringkas = led.drop_duplicates("kode")[[
    "kode", "faktor", "teks", "gambar", "fusion", "f1_macro", "acc", "bal_acc", "params"
]]
display(ringkas.sort_values(["faktor", "kode"]))

test_files = sorted(RUNS_DIR.glob("Z*-test.npz"))
if test_files:
    test_eval = test.groupby("label").head(35) if SMOKE else test
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
