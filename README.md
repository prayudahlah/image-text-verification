# image-text-verification

Verifikasi keselarasan gambar dan teks berbahasa Indonesia. Diberi satu gambar dan satu
caption, sistem menentukan apakah keduanya cocok (`true`), caption-nya salah
menggambarkan gambar (`miscaptioned`), atau gambar berasal dari konteks lain
(`out-of-context`).

## Struktur

```
.
├── data_collection/
│   └── 01_pipeline_dataset.ipynb   pengumpulan dan penyusunan dataset
├── data/                            pasangan gambar dan teks (hasil pipeline)
├── notebooks/                       satu notebook untuk pemodelan
└── reports/                         gambar dan tabel untuk laporan
```

## Dataset

| Berkas | Isi |
|---|---|
| `data/01_urls.json` | daftar URL sumber |
| `data/02_articles_raw.jsonl` | hasil parsing mentah, dipakai sebagai cache |
| `data/03_articles.jsonl` | korpus setelah penyaringan |
| `data/03b_articles_en.jsonl` | artikel yang dibuang karena caption berbahasa Inggris |
| `data/04_images.jsonl` | manifest unduhan: URL, berkas, dimensi, status |
| `data/images/` | berkas gambar pasangan, tidak ikut di-commit |
| `data/05_mc_raw.jsonl` | respons mentah model untuk caption miscaptioned |
| `data/06_pairs_no_folds.jsonl` | dataset sebelum pembagian fold |
| `data/07_pairs.jsonl` | dataset final dengan kolom `fold` |

Dataset terdiri dari 5.689 pasangan `true`, 5.689 `out-of-context`, dan 2.296
`miscaptioned`, total 13.674 pasangan dari 5.689 artikel. Artikel yang memakai foto sama
digabung berdasarkan path foto, bukan URL penuh.

Split dilakukan per artikel. Artikel dengan caption identik disatukan, dan pasangan
out-of-context dibentuk di dalam fold, sehingga tidak ada gambar atau caption yang
menyeberang antara latih dan uji. Set uji berisi 700 pasangan dari 278 artikel, sisanya
dibagi lima fold.

## Menjalankan

```bash
pip install -r requirements.txt
cp .env.example .env        # isi OPENCODE_API_KEY
jupyter lab data_collection/01_pipeline_dataset.ipynb
```

Notebook memanggil model hanya untuk artikel yang belum punya respons di
`data/05_mc_raw.jsonl`. Karena berkas itu ikut di repo, menjalankan ulang notebook tidak
memerlukan kunci API dan menghasilkan dataset yang sama.

Notebook pipeline memakai parameter `LIMIT` di sel kedua. Nilai 50 untuk uji cepat,
3900 untuk menjalankan penuh. Semua keluaran ditulis ke `data/`.

Daftar URL dibekukan di `data/01_urls.json`. Selama berkas itu ada dan `LIMIT` tidak
diubah, dua kali jalan penuh menghasilkan berkas yang identik; sidik jari `sha256`
dicetak di sel terakhir. Setel `RECOLLECT = True` hanya kalau ingin memperbarui korpus.

## Eksperimen fusion

Notebook pemodelan ada di `notebooks/03_eksperimen_fusion_v2.py`, format jupytext.
Sumbernya di-commit sebagai `.py`, lalu dibangun dan dieksekusi menjadi `.ipynb` oleh
runner. Versi pertama ada di `notebooks/02_eksperimen_fusion_v1.py` dan arsipnya di
`experiments/v1/`, dipakai sebagai pilot pemilihan protokol.

```bash
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python -r requirements-train.txt
.venv-train/bin/python -m ipykernel install --user --name image-text-fusion

SMOKE=1 .venv-train/bin/python experiments/run_notebook.py
.venv-train/bin/python experiments/run_notebook.py
```

Tahap dijalankan lewat variabel lingkungan `RUN_BASELINE`, `RUN_GRID`, `RUN_FUSION`,
`RUN_TUNING`, `RUN_OOF`, dan `RUN_FINAL`. Run yang sudah ada di `experiments/ledger.csv`
dilewati, kecuali `FORCE=1`. Rincian serah terima ada di `experiments/HANDOFF.md`.

## Lisensi dan sumber data

Korpus diambil dari portal berita publik untuk keperluan penelitian. Teks dan gambar
tetap milik penerbit asal dan tidak didistribusikan ulang di repositori ini; yang
disimpan hanya metadata dan pasangan teks yang diperlukan untuk reproduksi.
