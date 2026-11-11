# Data

`pairs.csv` berisi satu baris per pasangan gambar dan teks.

| Kolom | Isi |
|---|---|
| `id` | Nomor unik pasangan |
| `image` | Nama file gambar di `data/images/` |
| `caption` | Teks yang dipasangkan dengan gambar |
| `label` | `true`, `miscaptioned`, atau `out-of-context` |
| `source` | URL artikel asal |

Gambar tidak disimpan di repositori. Untuk mengunduh ulang, jalankan `data_collection/download_images.py` setelah `pairs.csv` tersedia.

## Label

- `true`, gambar dan caption berasal dari peristiwa yang sama.
- `miscaptioned`, gambar asli dengan klaim palsu yang beredar.
- `out-of-context`, caption benar tapi dipasangkan dengan gambar dari peristiwa lain.

## Lisensi

Gambar dan teks mengikuti lisensi sumber masing-masing. Jangan mendistribusikan ulang gambarnya di repositori publik.
