> Catatan: nama berkas di laporan ini mengikuti penamaan lama. Sejak 19 November 2026, berkas di `data/` diberi nomor urut, misalnya `pairs.jsonl` menjadi `08_pairs.jsonl`.

# Kerja semalam: 17–18 November 2026

Dijalankan otomatis atas persetujuan pengguna, tanpa commit dan tanpa push.
Target: minimal 6.000 pasangan dengan tiga label.

## Hasil

| Ukuran | Nilai |
|---|---|
| Total pasangan | **15.064** |
| true | 6.624 |
| out-of-context | 6.624 |
| miscaptioned | 1.816 |
| Fold | 3.013 / 3.012 / 3.013 / 3.013 / 3.013 |
| Kebocoran URL | 0 |
| Kebocoran foto | 0 |
| Kebocoran caption | 0 |
| Artikel | 6.624 |
| Gambar terunduh | 6.621 dari 6.624 |
| Biaya setara | 2,35 dolar (1,92 MC + 0,43 verifikasi) |

Target terlampaui: 15.064 pasangan, tiga label lengkap, `miscaptioned` 1.816.

## Yang dijalankan

1. **Pengambilan URL**: 56 tag × halaman 1–257 langkah 8, total 1.782 halaman, 13.362 kandidat URL, dipotong ke 5.000 setelah pembatasan 15 persen per bulan.
2. **Parsing**: 4.853 halaman, ditambah artikel lama yang sudah ada di `articles_raw.jsonl`. Total artikel unik setelah filter: 6.624.
3. **Deduplikasi foto**: 1.120 foto ganda dibuang, memakai kunci `/data/photo/...`.
4. **Unduh gambar**: 6.621 berhasil, 3 gagal.
5. **Verifikasi VLM** pada 6.624 pasangan `true`, dengan model yang sama.
6. **Generasi miscaptioned**: 132 batch untuk 1.056 kandidat baru, lalu 86 batch variasi.
7. **Penyusunan dan pembagian data**: 15.064 pasangan, lima fold seimbang, nol kebocoran.

## Temuan penting: gerbang VLM tidak layak dipakai

Diuji dengan 30 pasangan negatif (sudah pasti tidak cocok, diambil dari `mc_raw` valid) dan 30 positif:

| Kelompok | Dinilai cocok oleh VLM |
|---|---|
| Negatif (seharusnya ditolak) | **25 dari 30 (83 persen)** |
| Positif | 29 dari 30 (97 persen) |

Artinya model hanya menolak 5 dari 30 pasangan yang jelas salah. Gerbang ini hampir tidak membedakan, sehingga penanda `verified` **tidak boleh diklaim sebagai validasi**.

Yang tetap berlaku: penanda itu hanya menunjukkan "model menilai cocok", bukan "pasangan ini benar".

## Sebaran bulan

| Bulan | Pasangan | Porsi |
|---|---|---|
| 2026-11 | 3.857 | 25,6% |
| 2026-10 | 1.719 | 11,4% |
| 2026-09 | 1.189 | 7,9% |
| 2026-08 | 630 | 4,2% |
| bulan lain (102 bulan) | 7.669 | 50,9% |

Rentang kini mencakup 106 bulan, jauh lebih beragam daripada sebelumnya yang 93 persen dalam tiga bulan. Namun November masih 25,6 persen, karena korpus lama ikut terbawa.

Jarak unggah foto: median 0 hari, 16 persen lebih dari 90 hari.

## Keputusan yang kuambil sendiri

| Keputusan | Alasan |
|---|---|
| Memakai seluruh `articles_raw.jsonl`, termasuk artikel lama | mempercepat, dan artikel lama tetap memenuhi aturan |
| Tidak membuang pasangan yang ditolak VLM | gerbangnya terbukti lemah, membuangnya tidak berdasar |
| Menyimpan `verified` sebagai penanda, bukan filter | sama |
| Tidak mengubah definisi label | sesuai instruksi |

## Yang perlu diputuskan pagi

1. **Apakah 15.064 pasangan ini dipakai, atau dipangkas?** Dengan 106 bulan, distribusinya lebar, tapi korpus lama masih dominan di November.
2. **Apakah `verified` dibuang seluruhnya?** Saranku: dihapus dari laporan, atau diganti nama menjadi `vlm_opinion` supaya tidak disalahartikan.
3. **Audit manusia tetap perlu.** Tidak ada pengganti untuk itu, dan gerbang VLM sudah terbukti tidak bisa menggantikannya.

## Berkas

| Berkas | Isi |
|---|---|
| `data/pairs.jsonl` | 15.064 pasangan final dengan `fold`, `verified`, `photo_gap_days` |
| `data/mc_raw.jsonl` | 3.362 respons, termasuk yang ditolak |
| `data/vlm_check.jsonl` | 6.624 penilaian, 6.120 "ya", 442 "tidak", 62 gagal |
| `data/vlm_control.jsonl` | 60 baris kontrol negatif dan positif |
| `data/images/` | 6.621 gambar |

Repo tidak di-commit dan tidak di-push, sesuai permintaan.

## Pemeriksaan akhir

| Pemeriksaan | Hasil |
|---|---|
| Gambar hilang dari 6.624 yang dirujuk | 0 |
| Kebocoran teks (TF-IDF, true vs MC) | akurasi 0,697, AUC 0,744 |
| `urls.json` disamakan dengan sumber korpus | 6.624 URL |

Kebocoran teks 0,697 sedikit lebih rendah daripada dataset sebelumnya (0,718), kemungkinan karena
keragaman peristiwa meningkat dan kata pengganti jadi kurang dominan.

## Catatan teknis

- Notebook berisi 39 sel dan lolos validasi nbformat.
- Sel unduh gambar muncul dua kali; yang kedua hanya melewati berkas yang sudah ada.
- `verified` berasal dari jawaban model yang sama dengan penulis caption, dan kontrol
  menunjukkan daya pisah yang lemah.
