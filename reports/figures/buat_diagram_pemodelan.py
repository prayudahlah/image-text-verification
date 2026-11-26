"""Diagram alur pengembangan sistem multimodal untuk kesesuaian gambar & caption."""

from pathlib import Path
import shutil
import subprocess

OUT = Path("reports/figures").resolve()
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1460, 740
BIRU = "#2563eb"
BIRU_MUDA = "#eff6ff"
ABU = "#71717a"
GARIS = "#d4d4d8"
ISI = "#fafafa"
TEKS = "#18181b"
SUBTEKS = "#3f3f46"

langkah = [
    ("1", "Pengumpulan & dataset", "Scraping berita, gambar & caption", "pairing · LLM · labeling"),
    ("2", "Exploratory Data Analysis", "Distribusi label, kategori berita", "panjang caption · karakteristik"),
    ("3", "Preprocessing data", "Pembersihan teks, tokenisasi", "IndoBERT · mBERT · gambar"),
    ("4", "Pembagian & validasi", "Pool & test, validasi eksplorasi", "data leakage · 5-fold CV"),
    ("5", "Pemodelan & pembanding", "Baseline, model teks & gambar", "strategi multimodal fusion"),
    ("6", "Arsitektur & tuning", "Metode fusion, kombinasi encoder", "learning rate · dropout"),
    ("7", "Pelatihan & evaluasi", "5-fold OOF, tiga model final", "pengujian pada data test"),
    ("8", "Hasil & error analysis", "Evaluasi metrik, confusion matrix", "kesalahan · generalisasi"),
]

langkah_9 = (
    "9",
    "Deployment sistem",
    "Integrasi model terpilih ke aplikasi untuk prediksi kesesuaian gambar dan caption",
    "inference · kesesuaian gambar & caption",
)

BW, BH = 300, 122
ROW1, ROW2 = 126, 326
ROW3_Y = 516
X = [24, 384, 744, 1104]

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'font-family="Inter, system-ui, sans-serif">'
]
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
svg.append(
    f'<text x="24" y="52" font-size="25" font-weight="700" fill="{TEKS}">'
    f"Alur pengembangan sistem multimodal</text>"
)
svg.append(
    f'<text x="24" y="82" font-size="14" fill="{ABU}">'
    f"Dari pembentukan dataset, eksperimen multimodal, evaluasi, hingga deployment prediksi gambar &amp; caption</text>"
)


def kotak(x, y, w, h, nomor, judul, baris1, baris2, highlight=False):
    bg = BIRU_MUDA if highlight else ISI
    stroke = BIRU if highlight else GARIS
    sw = "1.8" if highlight else "1.5"
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
        f'fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    svg.append(f'<circle cx="{x + 22}" cy="{y + 24}" r="12" fill="{BIRU}"/>')
    svg.append(
        f'<text x="{x + 22}" y="{y + 29}" font-size="12" font-weight="700" '
        f'fill="#ffffff" text-anchor="middle">{nomor}</text>'
    )
    svg.append(
        f'<text x="{x + 44}" y="{y + 29}" font-size="15" font-weight="600" '
        f'fill="{TEKS}">{judul}</text>'
    )
    svg.append(
        f'<text x="{x + 16}" y="{y + 62}" font-size="12.5" fill="{SUBTEKS}">{baris1}</text>'
    )
    svg.append(
        f'<text x="{x + 16}" y="{y + 92}" font-size="11.5" fill="{ABU}" '
        f'font-family="ui-monospace, monospace">{baris2}</text>'
    )


for i, (n, j, b1, b2) in enumerate(langkah[:4]):
    kotak(X[i], ROW1, BW, BH, n, j, b1, b2)

for i, (n, j, b1, b2) in enumerate(langkah[4:]):
    kotak(X[3 - i], ROW2, BW, BH, n, j, b1, b2)


def panah_horizontal(x1, y1, x2, y2, arah="kanan"):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BIRU}" stroke-width="2"/>')
    if arah == "kanan":
        svg.append(f'<path d="M {x2} {y2} l -10 -5 l 0 10 z" fill="{BIRU}"/>')
    else:
        svg.append(f'<path d="M {x2} {y2} l 10 -5 l 0 10 z" fill="{BIRU}"/>')


def panah_vertikal(x, y1, y2):
    svg.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 6}" stroke="{BIRU}" stroke-width="2"/>')
    svg.append(f'<path d="M {x} {y2} l -5 -9 l 10 0 z" fill="{BIRU}"/>')


for i in range(3):
    panah_horizontal(X[i] + BW + 4, ROW1 + BH / 2, X[i + 1] - 6, ROW1 + BH / 2, arah="kanan")

panah_vertikal(X[3] + BW / 2, ROW1 + BH + 4, ROW2 - 2)

for i in range(3):
    panah_horizontal(X[3 - i] - 4, ROW2 + BH / 2, X[3 - i - 1] + BW + 6, ROW2 + BH / 2, arah="kiri")

panah_vertikal(X[0] + BW / 2, ROW2 + BH + 4, ROW3_Y - 2)

kotak(
    24,
    ROW3_Y,
    1412,
    BH,
    langkah_9[0],
    langkah_9[1],
    langkah_9[2],
    langkah_9[3],
    highlight=True,
)

svg.append(
    f'<rect x="24" y="654" width="1412" height="62" rx="10" '
    f'fill="{ISI}" stroke="{GARIS}" stroke-width="1.2"/>'
)
svg.append(
    f'<text x="44" y="680" font-size="13" font-weight="600" fill="{TEKS}">Ringkasan alur:</text>'
)
svg.append(
    f'<text x="152" y="680" font-size="13" fill="{SUBTEKS}">'
    f"1–4 Tahap data &amp; validasi · 5–7 Eksperimen, tuning &amp; evaluasi · 8–9 Analisis kesalahan &amp; deployment</text>"
)
svg.append(
    f'<text x="44" y="702" font-size="12" fill="{ABU}" font-family="ui-monospace, monospace">'
    f"Output akhir: model multimodal siap pakai untuk verifikasi kesesuaian gambar &amp; caption</text>"
)

svg.append("</svg>")

svg_content = "".join(svg)
svg_path = OUT / "pemodelan_pipeline.svg"
png_path = OUT / "pemodelan_pipeline.png"
html_path = OUT / "pemodelan_pipeline.html"

svg_path.write_text(svg_content, encoding="utf-8")
html_path.write_text(
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<style>html,body{margin:0;padding:12px;background:#fff;overflow-x:hidden;}"
    "svg{max-width:100%;height:auto;display:block;}</style></head>"
    f"<body>{svg_content}</body></html>",
    encoding="utf-8",
)


def render_png_pillow(out_png: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    s = 2
    img = Image.new("RGB", (W * s, H * s), "#ffffff")
    draw = ImageDraw.Draw(img)

    def fnt(size, bold=False, mono=False):
        kandidat = (
            ["consola.ttf", "lucon.ttf", "DejaVuSansMono.ttf"]
            if mono
            else (
                ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"]
                if bold
                else ["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
            )
        )
        for nama in kandidat:
            for folder in [Path(r"C:\Windows\Fonts"), Path("/usr/share/fonts")]:
                p = folder / nama
                if p.exists():
                    try:
                        return ImageFont.truetype(str(p), size * s)
                    except OSError:
                        pass
        return ImageFont.load_default()

    draw.text((24 * s, 28 * s), "Alur pengembangan sistem multimodal", fill=TEKS, font=fnt(25, bold=True))
    draw.text(
        (24 * s, 64 * s),
        "Dari pembentukan dataset, eksperimen multimodal, evaluasi, hingga deployment prediksi gambar & caption",
        fill=ABU,
        font=fnt(14),
    )

    def gambar_kotak(x, y, w, h, nomor, judul, b1, b2, highlight=False):
        bg = BIRU_MUDA if highlight else ISI
        border = BIRU if highlight else GARIS
        draw.rounded_rectangle(
            [x * s, y * s, (x + w) * s, (y + h) * s],
            radius=12 * s,
            fill=bg,
            outline=border,
            width=2 * s,
        )
        draw.ellipse([(x + 10) * s, (y + 12) * s, (x + 34) * s, (y + 36) * s], fill=BIRU)
        draw.text(((x + 18.5) * s, (y + 15) * s), nomor, fill="#ffffff", font=fnt(12, bold=True))
        draw.text(((x + 44) * s, (y + 14) * s), judul, fill=TEKS, font=fnt(15, bold=True))
        draw.text(((x + 16) * s, (y + 50) * s), b1, fill=SUBTEKS, font=fnt(12))
        draw.text(((x + 16) * s, (y + 82) * s), b2, fill=ABU, font=fnt(11, mono=True))

    for i, (n, j, b1, b2) in enumerate(langkah[:4]):
        gambar_kotak(X[i], ROW1, BW, BH, n, j, b1, b2)
    for i, (n, j, b1, b2) in enumerate(langkah[4:]):
        gambar_kotak(X[3 - i], ROW2, BW, BH, n, j, b1, b2)
    gambar_kotak(24, ROW3_Y, 1412, BH, *langkah_9, highlight=True)

    def panah_h(x1, y1, x2, y2, arah="kanan"):
        draw.line([(x1 * s, y1 * s), (x2 * s, y2 * s)], fill=BIRU, width=2 * s)
        if arah == "kanan":
            pts = [(x2 * s, y2 * s), ((x2 - 10) * s, (y2 - 5) * s), ((x2 - 10) * s, (y2 + 5) * s)]
        else:
            pts = [(x2 * s, y2 * s), ((x2 + 10) * s, (y2 - 5) * s), ((x2 + 10) * s, (y2 + 5) * s)]
        draw.polygon(pts, fill=BIRU)

    def panah_v(x, y1, y2):
        draw.line([(x * s, y1 * s), (x * s, (y2 - 6) * s)], fill=BIRU, width=2 * s)
        pts = [(x * s, y2 * s), ((x - 5) * s, (y2 - 9) * s), ((x + 5) * s, (y2 - 9) * s)]
        draw.polygon(pts, fill=BIRU)

    for i in range(3):
        panah_h(X[i] + BW + 4, ROW1 + BH // 2, X[i + 1] - 6, ROW1 + BH // 2, "kanan")
    panah_v(X[3] + BW // 2, ROW1 + BH + 4, ROW2 - 2)
    for i in range(3):
        panah_h(X[3 - i] - 4, ROW2 + BH // 2, X[3 - i - 1] + BW + 6, ROW2 + BH // 2, "kiri")
    panah_v(X[0] + BW // 2, ROW2 + BH + 4, ROW3_Y - 2)

    draw.rounded_rectangle(
        [24 * s, 654 * s, 1436 * s, 716 * s],
        radius=10 * s,
        fill=ISI,
        outline=GARIS,
        width=2,
    )
    draw.text((44 * s, 668 * s), "Ringkasan alur:", fill=TEKS, font=fnt(13, bold=True))
    draw.text(
        (152 * s, 668 * s),
        "1–4 Tahap data & validasi · 5–7 Eksperimen, tuning & evaluasi · 8–9 Analisis kesalahan & deployment",
        fill=SUBTEKS,
        font=fnt(13),
    )
    draw.text(
        (44 * s, 692 * s),
        "Output akhir: model multimodal siap pakai untuk verifikasi kesesuaian gambar & caption",
        fill=ABU,
        font=fnt(12, mono=True),
    )

    img.save(out_png)
    return out_png.exists()


def simpan_png(svg_bytes: bytes, out_png: Path) -> bool:
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(out_png), scale=2.0)
        return True
    except Exception:
        pass

    if render_png_pillow(out_png):
        return True

    for exe in ("msedge", "chrome", "google-chrome", "chromium"):
        browser = shutil.which(exe)
        if browser:
            cmd = [
                browser,
                "--headless",
                "--disable-gpu",
                f"--screenshot={out_png}",
                f"--window-size={W},{H}",
                html_path.as_uri(),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and out_png.exists():
                return True

    return False


print("ditulis:", svg_path)
print("ditulis:", html_path)
if simpan_png(svg_content.encode("utf-8"), png_path):
    print("ditulis:", png_path)
else:
    print("gagal membuat PNG otomatis.")