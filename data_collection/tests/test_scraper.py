"""
Unit Tests for Indonesian News Scraper
Tests extraction, factcheck validation, image validation, deduplication, and storage.
"""

import io
import json
from pathlib import Path
import pytest
from PIL import Image
from bs4 import BeautifulSoup

from processors.image_validator import ImageValidator
from processors.factcheck_validator import FactCheckValidator
from processors.article_extractor import ArticleExtractor
from processors.deduplicator import Deduplicator
from storage.database import Database
from storage.exporter import Exporter


# ==========================================
# 1. IMAGE VALIDATOR TESTS
# ==========================================
def test_image_url_blacklist():
    validator = ImageValidator()
    
    # Tracking pixels and logos should be rejected
    is_bad, _ = validator.is_url_blacklisted("https://example.com/assets/logo-mobile.png")
    assert is_bad is True

    is_bad, _ = validator.is_url_blacklisted("https://example.com/tracker/1x1.gif")
    assert is_bad is True

    # Real news image URL should pass blacklist
    is_bad, _ = validator.is_url_blacklisted("https://asset.kompas.com/crops/news_photo_1200x800.jpg")
    assert is_bad is False


def test_pillow_image_dimension_validation(tmp_path):
    validator = ImageValidator(download_dir=tmp_path)

    # 1. Test image smaller than minimum required resolution (e.g. 50x50)
    small_img = Image.new("RGB", (50, 50), color="blue")
    small_bytes = io.BytesIO()
    small_img.save(small_bytes, format="JPEG")
    small_bytes.seek(0)

    # Create dummy mock session
    class MockResp:
        status_code = 200
        headers = {"Content-Type": "image/jpeg"}
        def iter_content(self, chunk_size):
            yield small_bytes.getvalue()

    validator.session.get = lambda *args, **kwargs: MockResp()
    is_valid, reason, _ = validator.validate_and_save("https://example.com/small.jpg")
    assert is_valid is False
    assert "below minimum required" in reason

    # 2. Test valid high resolution image (600x400)
    valid_img = Image.new("RGB", (600, 400), color="green")
    valid_bytes = io.BytesIO()
    valid_img.save(valid_bytes, format="JPEG")
    valid_bytes.seek(0)

    class ValidMockResp:
        status_code = 200
        headers = {"Content-Type": "image/jpeg"}
        def iter_content(self, chunk_size):
            yield valid_bytes.getvalue()

    validator.session.get = lambda *args, **kwargs: ValidMockResp()
    is_valid, reason, local_path = validator.validate_and_save("https://example.com/valid.jpg", save_locally=True)
    assert is_valid is True
    assert reason is None
    assert local_path is not None
    assert Path(local_path).exists()


# ==========================================
# 2. FACT CHECK VALIDATOR TESTS
# ==========================================
def test_factcheck_tag_and_status():
    validator = FactCheckValidator()

    # Case 1: [SALAH] tag -> FALSE
    res = validator.validate(
        headline="Kapolri Copot Kapolda Jateng Buntut Penggerebekan",
        article_url="https://turnbackhoax.id/articles/123",
        source_name="TurnBackHoax",
        raw_headline="[SALAH] Kapolri Copot Kapolda Jateng Buntut Penggerebekan"
    )
    assert res["verification_type"] == "DEDICATED_FACT_CHECK"
    assert res["fact_check_status"] == "FALSE"
    assert "Kapolri Copot Kapolda Jateng" in res["claim_checked"]
    assert res["fact_check_source"] == "TurnBackHoax (MAFINDO)"

    # Case 2: [KLARIFIKASI] tag -> MISLEADING
    res_klarifikasi = validator.validate(
        headline="Foto Gibran Jalani Pemeriksaan di Mabes TNI",
        article_url="https://www.kompas.com/cekfakta/read/2026/11/10/foto-gibran",
        source_name="Kompas",
        raw_headline="[KLARIFIKASI] Foto Gibran Jalani Pemeriksaan di Mabes TNI"
    )
    assert res_klarifikasi["verification_type"] == "DEDICATED_FACT_CHECK"
    assert res_klarifikasi["fact_check_status"] == "MISLEADING"
    assert "Foto Gibran Jalani Pemeriksaan" in res_klarifikasi["claim_checked"]
    assert res_klarifikasi["fact_check_source"] == "Cek Fakta Kompas"

    # Case 3: Regular news with primary source attribution -> VERIFIED_NEWS
    res_news = validator.validate(
        headline="Pemerintah Salurkan Bantuan Beras untuk Korban Bencana",
        article_url="https://nasional.tempo.co/read/2123221/pemerintah-salurkan-bantuan",
        source_name="Tempo",
        body_text="Bantuan diserahkan secara simbolis. Menurut keterangan resmi Menteri Sosial, program ini menyasar 500 kepala keluarga."
    )
    assert res_news["verification_type"] == "VERIFIED_NEWS"
    assert res_news["fact_check_status"] == "NOT_CHECKED"
    assert res_news["claim_checked"] is None
    assert res_news["fact_check_source"] is None


# ==========================================
# 3. ARTICLE EXTRACTOR TESTS
# ==========================================
def test_headline_branding_cleanup():
    cases = [
        ("Gibran Hadiri Rakornas di Jakarta - Kompas.com", "Gibran Hadiri Rakornas di Jakarta"),
        ("Polisi Tangkap Pengedar Narkoba - DetikNews", "Polisi Tangkap Pengedar Narkoba"),
        ("Daftar Harga BBM Terbaru | Tempo", "Daftar Harga BBM Terbaru"),
        ("Pemerintah Perkuat Kerjasama - ANTARA News", "Pemerintah Perkuat Kerjasama"),
        ("Kumpulan Hoaks Catut Tokoh - Liputan6.com", "Kumpulan Hoaks Catut Tokoh"),
        ("Perkembangan Kasus Hukum Terkini - BBC News Indonesia", "Perkembangan Kasus Hukum Terkini")
    ]
    for raw, expected in cases:
        assert ArticleExtractor.clean_headline(raw) == expected


def test_caption_concise_generation():
    body = (
        "Jakarta, Kompas - Pemerintah hari ini mengumumkan paket kebijakan ekonomi baru. "
        "Kebijakan ini bertujuan untuk menstabilkan harga komoditas pokok menjelang akhir tahun. "
        "Menteri Keuangan menyatakan langkah fiskal ini telah dikoordinasikan dengan Bank Indonesia. "
        "Selain itu, subsidi energi tetap dipertahankan sesuai anggaran APBN 2026."
    )
    caption = ArticleExtractor.generate_concise_caption(body, max_sentences=3)
    assert caption is not None
    # Ensure it's 3 sentences
    assert caption.count(".") >= 2
    assert "Pemerintah hari ini mengumumkan" in caption


# ==========================================
# 4. DEDUPLICATOR TESTS
# ==========================================
def test_url_canonicalization_and_deduplication():
    dedup = Deduplicator(similarity_threshold=0.8)

    url1 = "https://www.kompas.com/cekfakta/read/12345?utm_source=facebook&utm_medium=social#comment"
    url2 = "https://www.kompas.com/cekfakta/read/12345"

    assert dedup.normalize_url(url1) == "https://www.kompas.com/cekfakta/read/12345"
    assert dedup.normalize_url(url2) == "https://www.kompas.com/cekfakta/read/12345"

    dedup.register(url1, "Kabar Hoaks Gempa Bumi Megathrust")
    
    # Same canonical URL
    is_dup, reason = dedup.is_duplicate(url2, "Judul Berbeda Sedikit")
    assert is_dup is True
    assert "Canonical URL already scraped" in reason

    # Similar headline with different URL
    is_dup2, reason2 = dedup.is_duplicate("https://other.com/news/1", "Kabar Hoaks Gempa Bumi Megathrust di Jawa")
    assert is_dup2 is True
    assert "Headline similarity" in reason2


# ==========================================
# 5. DATABASE AND EXPORTER TESTS
# ==========================================
def test_database_and_exporter(tmp_path):
    db_file = tmp_path / "test_news.db"
    json_file = tmp_path / "test_news.json"
    csv_file = tmp_path / "test_news.csv"

    db = Database(db_file)
    exporter = Exporter(json_file, csv_file)

    sample_article = {
        "thumbnail_url": "https://example.com/image.jpg",
        "headline": "Klaim Vaksin Mengandung Magnet Adalah Hoaks",
        "main_caption": "Beredar video yang mengeklaim vaksin menempel magnet. Hal tersebut dipastikan hoaks oleh dokter.",
        "source_name": "Kompas",
        "article_url": "https://www.kompas.com/cekfakta/read/9999",
        "publication_date": "2026-11-10",
        "verification_type": "DEDICATED_FACT_CHECK",
        "fact_check_status": "FALSE",
        "fact_check_source": "Cek Fakta Kompas",
        "claim_checked": "Vaksin mengandung kepingan magnet",
        "verification_url": "https://www.kompas.com/cekfakta/read/9999"
    }

    row_id = db.save_article(sample_article)
    assert row_id is not None
    assert db.count_articles() == 1
    assert db.article_exists(sample_article["article_url"]) is True

    # Test rejection of non-fact-checked article
    non_factcheck = {
        "thumbnail_url": "https://example.com/image2.jpg",
        "headline": "Berita Biasa Tanpa Cek Fakta",
        "main_caption": "Keterangan berita biasa.",
        "source_name": "Kompas",
        "article_url": "https://www.kompas.com/read/8888",
        "publication_date": "2026-11-10",
        "verification_type": "VERIFIED_NEWS",
        "fact_check_status": "NOT_CHECKED"
    }
    rejected_id = db.save_article(non_factcheck)
    assert rejected_id is None, "Non-factcheck article must be rejected by Database.save_article"
    assert db.count_articles() == 1

    # Retrieve from DB
    articles = db.get_articles()
    assert len(articles) == 1
    assert articles[0]["headline"] == sample_article["headline"]

    # Export to JSON and CSV
    exporter.export_all(articles)
    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "articles" in data
        assert len(data["articles"]) == 1
        item = data["articles"][0]
        assert item["id"] == row_id
        assert item["thumbnail_url"] == sample_article["thumbnail_url"]
        assert item["headline"] == sample_article["headline"]
        assert item["fact_check_status"] == "FALSE"

