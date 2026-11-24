"""Diagram alur pipeline dataset."""

from pathlib import Path

OUT = Path("/home/prayudahlah/projects/college/ml/image-text-verification/reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1460, 660
BIRU = "#2563eb"
ABU = "#71717a"
GARIS = "#d4d4d8"
ISI = "#fafafa"

langkah = [
    ("1", "Kumpulkan URL", "54 topik, 33 halaman per topik", "01_urls.json · 6.624 URL"),
    ("2", "Parsing artikel", "judul, gambar, caption, channel", "02_articles_raw.jsonl · 8.848"),
    ("3", "Saring dan dedup", "field, foto ganda, caption Inggris", "03_articles.jsonl · 6.308"),
    ("4", "Bentuk pasangan", "true dan out-of-context", "6.308 + 6.308 pasangan"),
    ("5", "Tulis ulang MC", "ganti entitas dengan LLM", "05_mc_raw.jsonl · 1.648 dipakai"),
    ("6", "Susun dataset", "tiga kelas, tanpa kebocoran", "06_pairs_no_folds.jsonl · 14.264"),
    ("7", "Pilih test, bagi fold", "test per kelompok, lalu lima fold", "test 701 · pool 13.563"),
    ("8", "Unduh gambar", "nama dari hash URL, cek ukuran", "04_images.jsonl · 6.308"),
]

BW, BH = 300, 112
ROW1, ROW2 = 128, 368
X = [24, 384, 744, 1104]

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'font-family="Inter, system-ui, sans-serif">']
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append('<text x="24" y="54" font-size="25" font-weight="700" fill="#18181b">Alur pipeline dataset</text>')
svg.append(f'<text x="24" y="84" font-size="14" fill="{ABU}">Dari halaman tag Kompas sampai dataset tiga kelas dengan set uji terpisah</text>')

def kotak(x, y, nomor, judul, isi, keluaran):
    svg.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="12" fill="{ISI}" stroke="{GARIS}" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{x+22}" cy="{y+22}" r="12" fill="{BIRU}"/>')
    svg.append(f'<text x="{x+22}" y="{y+27}" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">{nomor}</text>')
    svg.append(f'<text x="{x+44}" y="{y+27}" font-size="15" font-weight="600" fill="#18181b">{judul}</text>')
    svg.append(f'<text x="{x+16}" y="{y+58}" font-size="12" fill="#3f3f46">{isi}</text>')
    svg.append(f'<text x="{x+16}" y="{y+85}" font-size="11" fill="{ABU}" font-family="ui-monospace, monospace">{keluaran}</text>')

for i, (n, j, isi, kel) in enumerate(langkah[:4]):
    kotak(X[i], ROW1, n, j, isi, kel)
for i, (n, j, isi, kel) in enumerate(langkah[4:]):
    kotak(X[3 - i], ROW2, n, j, isi, kel)

def panah(x1, y1, x2, y2, arah="kanan"):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BIRU}" stroke-width="2"/>')
    if arah == "kanan":
        svg.append(f'<path d="M {x2} {y2} l -10 -5 l 0 10 z" fill="{BIRU}"/>')
    else:
        svg.append(f'<path d="M {x2} {y2} l 10 -5 l 0 10 z" fill="{BIRU}"/>')

for i in range(3):
    panah(X[i] + BW + 4, ROW1 + BH / 2, X[i + 1] - 6, ROW1 + BH / 2)

x_turun = X[3] + BW / 2
svg.append(f'<line x1="{x_turun}" y1="{ROW1+BH}" x2="{x_turun}" y2="{ROW2-8}" stroke="{BIRU}" stroke-width="2"/>')
svg.append(f'<path d="M {x_turun} {ROW2-2} l -5 -9 l 10 0 z" fill="{BIRU}"/>')

for i in range(3):
    panah(X[3 - i] - 4, ROW2 + BH / 2, X[3 - i - 1] + BW + 6, ROW2 + BH / 2, arah="kiri")

svg.append(f'<rect x="24" y="532" width="1412" height="98" rx="12" fill="#eff6ff" stroke="{BIRU}" stroke-width="1.2"/>')
svg.append('<text x="44" y="562" font-size="14" font-weight="600" fill="#18181b">Dataset akhir</text>')
svg.append('<text x="44" y="588" font-size="13" fill="#3f3f46">14.264 pasangan · true 6.308 · out-of-context 6.308 · miscaptioned 1.648</text>')
svg.append('<text x="44" y="612" font-size="13" fill="#3f3f46">Set uji 701 pasangan dari 310 artikel, terpisah per kelompok · lima fold di 13.563 pasangan · nol kebocoran</text>')
svg.append(f'<text x="1108" y="566" font-size="12" fill="{ABU}">Gambar: 6.308 berkas</text>')
svg.append(f'<text x="1108" y="588" font-size="12" fill="{ABU}">di data/images/</text>')
svg.append(f'<text x="1108" y="610" font-size="12" fill="{ABU}">04_images.jsonl</text>')
svg.append("</svg>")

(OUT / "dataset_pipeline.svg").write_text("".join(svg))
(OUT / "dataset_pipeline.html").write_text(
    '<html><body style="margin:0;background:#fff">' + "".join(svg) + "</body></html>")
print("ditulis:", OUT / "dataset_pipeline.svg")
