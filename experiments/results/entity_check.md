# Uji cek entitas per entitas (18 November 2026)

Model: DeepSeek V4.1 Flash, gambar 512 px. Uji pada 100 pasangan negatif (sudah pasti
tidak cocok) dan 100 positif.

## Cara

1. Ekstraksi entitas visual dari caption (maksimal 4).
2. Satu pertanyaan sempit per entitas, ditambah satu entitas jebakan yang pasti tidak ada.
3. Agregasi: porsi entitas yang dinilai terlihat.

## Hasil

| Ukuran | Nilai |
|---|---|
| Negatif tertangkap (minimal satu entitas tidak terlihat) | 88/100 |
| Positif dinilai cocok (semua entitas terlihat) | 27/100 |
| Jawaban jebakan yang keliru | 0/100 |

Jebakan tidak pernah dijawab "ya", jadi model tidak asal setuju. Tapi ia terlalu sering
menjawab "tidak".

## Kalau dipakai sebagai ambang

| Ambang coverage | Negatif tertangkap | Positif ikut tertolak |
|---|---|---|
| 0,00 | 19% | 11% |
| 0,34 | 44% | 28% |
| 0,50 | 67% | 48% |
| 0,67 | 84% | 67% |

Tidak ada titik yang layak. Pada ambang yang menangkap 67 persen negatif, hampir separuh
pasangan benar ikut tertolak.

## Kenapa gagal

- Entitas hasil ekstraksi sering tidak persis terlihat: "petugas pemadam" padahal foto
  hanya menampilkan punggung orang.
- Instruksi "jika ragu jawab tidak" membuat model konservatif.
- Gambar 512 px menghilangkan detail yang dibutuhkan untuk memastikan objek.

## Kesimpulan

Cek entitas per entitas, dengan model dan resolusi ini, tidak layak dipakai sebagai
filter. Ia juga tidak lebih baik daripada gate ya/tidak sebelumnya: keduanya memiliki
daya pisah yang lemah.

Yang tersisa untuk memastikan `true`: audit manusia. Filter otomatis hanya bisa dipakai
sebagai penanda, bukan penentu.
