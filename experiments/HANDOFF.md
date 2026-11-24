# Handoff eksperimen fusion v2

Dokumen ini untuk Darell. Isinya cara menjalankan, apa yang sudah dibekukan, dan apa yang
tidak boleh diubah.

Versi 1 dari notebook tetap ada di `notebooks/02_eksperimen_fusion_v1.ipynb` dan
`experiments/v1/`. Fungsinya sebagai pilot: bukti kenapa protokol latih diganti dan kenapa
M4 dipilih sebagai kombinasi acuan. Jangan dipakai untuk angka laporan.

## Lingkungan

```
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python -r requirements-train.txt
.venv-train/bin/python -m ipykernel install --user --name image-text-fusion
```

Kernel Jupyter bernama `image-text-fusion`. Hanya kernel itu yang punya torch, timm, dan
transformers.

## Menjalankan

Sumber notebook ada di `notebooks/03_eksperimen_fusion_v2.py`, format jupytext. Berkas
`.ipynb` dibangun dan dieksekusi oleh runner.

```
.venv-train/bin/python experiments/run_notebook.py
```

Mode smoke memakai 300 baris latih berstrata, satu epoch, dan ledger terpisah.

```
SMOKE=1 .venv-train/bin/python experiments/run_notebook.py
```

## Variabel lingkungan

| Variabel | Default | Isi |
|---|---|---|
| `RUN_BASELINE` | 1 | B0, A1, A2 |
| `RUN_FUSION` | 1 | F0 sampai F5 |
| `RUN_GRID` | 0 | M1 sampai M4 dengan fusion pemenang |
| `RUN_TUNING` | 0 | T1 sampai T6 |
| `RUN_OOF` | 0 | O1 sampai O5 |
| `RUN_FINAL` | 0 | Z1 sampai Z3 |
| `FUSION_ONLY` | kosong | jalankan satu varian fusion saja, misalnya `FUSION_ONLY=F1` |
| `SMOKE` | 0 | subset kecil, satu epoch |
| `FORCE` | 0 | ulang run walau sudah ada di ledger |

Contoh menjalankan tahap grid saja, setelah `pemenang_fusion.json` ada.

```
RUN_BASELINE=0 RUN_FUSION=0 RUN_GRID=1 .venv-train/bin/python experiments/run_notebook.py
```

Run yang sudah ada di ledger dilewati. Aman menjalankan ulang notebook tanpa mengulang
latihan.

## Protokol yang dibekukan

- Warmup 10 persen, cosine decay, maksimal 8 epoch, patience 3.
- Letterbox 224, `max_len` 128, tanpa augmentasi.
- Batch 16, AdamW, weight decay 0,01, clip grad 1,0, class weight.
- Matriks embedding kata dibekukan, lapisan transformer dilatih penuh.
- Bandingkan model memakai macro-F1 dan pairwise true lawan MC.
- Seed 727 untuk eksplorasi dan OOF, 727 sampai 729 untuk model final.

## Yang sudah diputuskan

| Bab | Keputusan | Berkas |
|---|---|---|
| 6 | enam varian fusion diuji pada kombinasi acuan M4 mBERT + DINOv2 | `experiments/ledger.csv` |
| 7 | fusion pemenang: **F3 cross-attention satu arah** | `experiments/pemenang_fusion.json` |
| 8 | kombinasi encoder pemenang | `experiments/pemenang.json` |

F2 gated dan F3 satu arah seri, selisih macro-F1 0,0058. Aturan tie-break dinyatakan
sebelum grid: kalau selisih di bawah 0,02, dipilih strategi yang menjadi fokus penelitian
selama skornya masih di dalam 0,02. Karena itu F3 yang dipakai, dan gated tetap dilaporkan
sebagai pembanding setara di Bab IV.

Bab 8 belum dijalankan saat serah terima. Darell mulai dari sana, memakai `xattn_uni`.

## Berkas yang dihasilkan

```
experiments/
  ledger.csv                  satu baris per run v2
  pemenang_fusion.json        hasil bab 7
  pemenang.json               hasil bab 8
  figures/                    grafik tiap bab
  runs/<run_id>.pt            checkpoint terbaik
  runs/<run_id>-val.npz       ids, y, prob validasi
  runs/<run_id>-test.npz      hanya model final
  runs/oof.npz                gabungan lima fold
  v1/                         arsip pilot, jangan dipakai untuk angka
  environment.txt             pip freeze
```

## Bagian Darell

| Bab | Isi | Run |
|---|---|---|
| 8 | grid M1 sampai M4 dengan fusion pemenang | 4 |
| 9 | grid learning rate dan dropout | 6 |
| 10 | 5-fold OOF, satu seed | 5 |
| 11 | tiga seed di seluruh pool, uji 700 | 3 |
| 12 | confusion matrix, contoh salah, ringkasan | 0 |

Urutannya berhenti di bab 9 kalau tuning tidak lebih baik dari konfigurasi dasar. Dalam
kasus itu pakai konfigurasi dari `pemenang.json` dan tulis alasannya di Bab IV.

## Metrik

- `f1_macro` metrik utama, karena kelas MC hanya 16,8 persen.
- `bal_acc` pembanding terhadap B0.
- Pairwise true lawan MC adalah metrik verifikasi. Text-only mencapai 0,778 pada pilot v1,
  jadi model fusion harus melampaui angka itu untuk klaim bahwa gambar dipakai.
- `f1_mc` jangan dibaca sendirian. Audit manusia menunjukkan sebagian item MC tidak bisa
  diperiksa dari gambar.
