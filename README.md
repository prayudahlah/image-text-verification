# image-text-verification

Verifikasi keselarasan gambar dan teks berbahasa Indonesia. Diberi satu gambar dan satu caption, model menentukan apakah keduanya cocok, caption-nya salah, atau gambarnya berasal dari konteks lain.

## Struktur

```
.
├── data_collection/     panen data dari sumber cek fakta
├── data/                pasangan gambar dan teks hasil panen
├── notebooks/           satu notebook untuk seluruh pemodelan
└── reports/             gambar dan tabel untuk laporan
```

## Menjalankan

```
pip install -r requirements.txt
jupyter lab notebooks/project.ipynb
```

## Data

`data/pairs.csv` berisi pasangan gambar dan teks beserta labelnya. Gambar tidak disimpan di repositori karena ukuran dan lisensinya. Cara mengunduh ulang dijelaskan di `data/README.md`.

## Lisensi

Kode berlisensi MIT. Data mengikuti lisensi sumber masing-masing dan tidak untuk didistribusikan ulang.
