# Handoff eksperimen fusion v2

Untuk Darell. Notebook ini diserahkan setengah jadi. Bab 1 sampai 7 sudah selesai dan
sudah dijalankan. Bab 8 sampai 12 sengaja kosong, hanya berisi markdown dan TODO, dan
kodenya kamu tulis sendiri.

## Pembagian

| Bab | Isi | Status |
|---|---|---|
| 1 sampai 4 | persiapan, data, preprocessing, split | selesai |
| 5 | B0, A1, A2 | selesai |
| 6 | enam varian fusion | selesai |
| 7 | analisis dan `pemenang_fusion.json` | selesai |
| 8 | grid encoder dengan fusion pemenang | TODO |
| 9 | tuning hyperparameter | TODO |
| 10 | 5-fold OOF | TODO |
| 11 | tiga seed dan uji 700 | TODO |
| 12 | analisis error | TODO |

## Lingkungan

```
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python -r requirements-train.txt
.venv-train/bin/python -m ipykernel install --user --name image-text-fusion
```

Kernel Jupyter bernama `image-text-fusion`. Hanya kernel itu yang punya torch, timm, dan
transformers.

## Cara melanjutkan

1. Buka `notebooks/03_eksperimen_fusion_v2.ipynb` di Jupyter, bukan berkas `.py`.
2. Jalankan sel dari atas sampai bab 7. Semua run dilewati karena ledger sudah berisi,
   jadi hanya butuh satu sampai dua menit. Gambar akan diunduh otomatis dari Drive pada
   bab 3 kalau belum ada.
3. Tulis dan jalankan sel bab 8 sampai 12. Markdown tiap bab berisi tujuan, input, output,
   keputusan, dan kontraknya.
4. Jangan jalankan `experiments/run_notebook.py` selama mengerjakan. Runner membangun ulang
   `.ipynb` dari `.py`, dan `.py` belum berisi kodemu, jadi pekerjaanmu akan hilang.
5. Setelah selesai, sinkronkan kembali ke `.py`:

```
jupytext --to py:percent notebooks/03_eksperimen_fusion_v2.ipynb -o notebooks/03_eksperimen_fusion_v2.py
```

## Data dan gambar

`data/07_pairs.jsonl` sudah ikut di repo, 13.674 baris. Notebook memeriksa jumlah itu
sebelum latih.

Gambar tidak ikut repo. Notebook mengunduh lima bagian arsip dari Drive, memverifikasi
sha256 tiap bagian, lalu mengekstraknya ke `data/images/`. Setelah gambar lengkap,
jalankan build cache sekali.

```
.venv-train/bin/python -m experiments.build_cache
```

## Helper yang tersedia

Semua sudah didefinisikan di bab 5 sampai 7.

| Nama | Fungsi |
|---|---|
| `Run(...)` | konfigurasi satu run, punya `run_id` |
| `run_one(run, train_rows, val_rows)` | latih, evaluasi, simpan checkpoint dan probabilitas, tulis ledger |
| `build_model(run)` | membangun model sesuai konfigurasi |
| `make_loader(rows, run, shuffle)` | dataloader |
| `pairwise_acc(run_id, pairs)` | pairwise true lawan MC dari berkas `-val.npz` |
| `train_rows`, `val_rows` | fold 1-4 dan fold 0 |
| `pool`, `test` | seluruh pool dan 700 test |
| `BASE` | hyperparameter dasar |

`run_one` melewati run yang `run_id`-nya sudah ada di ledger, dan mengembalikan baris
ledger yang sudah ada. Aman dipanggil berulang.

## Protokol yang dibekukan

- Warmup 10 persen, cosine, maksimal 8 epoch, patience 3.
- Letterbox 224, `max_len` 128, tanpa augmentasi.
- Batch 16, AdamW, weight decay 0,01, clip grad 1,0, class weight.
- Matriks embedding kata dibekukan, lapisan transformer dilatih penuh.
- Seed 727 untuk eksplorasi dan OOF, 727 sampai 729 untuk model final.

Jangan ubah protokol ini. Kalau ada yang perlu diubah, tulis alasannya di kolom `catatan`
ledger dan di commit message.

## Artefak yang harus dihasilkan

```
experiments/
  ledger.csv                  baris baru untuk setiap run
  pemenang.json               hasil bab 8
  figures/                    grafik bab 8 sampai 12
  runs/<run_id>.pt            checkpoint
  runs/<run_id>-val.npz       ids, y, prob validasi
  runs/<run_id>-test.npz      prediksi test, hanya model final
  runs/oof.npz                gabungan lima fold
```

## Metrik

- `f1_macro` metrik utama, karena kelas MC hanya 16,8 persen.
- `bal_acc` pembanding terhadap B0.
- Pairwise true lawan MC metrik verifikasi. Pada pilot, teks saja mencapai 0,810, jadi
  model fusion harus melampaui itu untuk klaim bahwa gambar dipakai.
- `f1_mc` jangan dibaca sendirian. Audit manusia menunjukkan sebagian item MC tidak bisa
  diperiksa dari gambar.
