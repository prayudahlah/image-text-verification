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
| `data/pairs_locked.jsonl` | pasangan final dengan kolom `fold` |
| `data/mc_v3_raw.jsonl` | respons mentah pembentukan caption dan alasan penolakan |
| `data/pairs_final.jsonl` | pasangan versi aturan, dipakai sebagai pembanding |

Split dilakukan per artikel. Artikel dengan caption identik disatukan, dan pasangan
out-of-context dibentuk di dalam fold, sehingga tidak ada gambar atau caption yang
menyeberang antara latih dan uji.

## Menjalankan

```bash
pip install -r requirements.txt
jupyter lab data_collection/01_pipeline_dataset.ipynb
```

Notebook pipeline memakai parameter `LIMIT` di sel kedua. Nilai 50 untuk uji cepat,
3900 untuk menjalankan penuh. Semua keluaran ditulis ke `data/`.

Daftar URL dibekukan di `data/urls.json`. Selama berkas itu ada dan `LIMIT` tidak
diubah, dua kali jalan penuh menghasilkan berkas yang identik; sidik jari `sha256`
dicetak di sel terakhir. Setel `RECOLLECT = True` hanya kalau ingin memperbarui korpus.

## Lisensi dan sumber data

Korpus diambil dari portal berita publik untuk keperluan penelitian. Teks dan gambar
tetap milik penerbit asal dan tidak didistribusikan ulang di repositori ini; yang
disimpan hanya metadata dan pasangan teks yang diperlukan untuk reproduksi.
