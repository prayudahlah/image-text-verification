# Configuration settings for Indonesian News Scraper
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
THUMBNAILS_DIR = OUTPUT_DIR / "thumbnails"
LOGS_DIR = BASE_DIR / "logs"
STORAGE_DIR = BASE_DIR / "storage"

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Database and Export paths
DB_PATH = STORAGE_DIR / "news.db"
JSON_OUTPUT_PATH = OUTPUT_DIR / "news.json"
CSV_OUTPUT_PATH = OUTPUT_DIR / "news.csv"
LOG_FILE_PATH = LOGS_DIR / "scraper.log"

# Network & Request Settings
REQUEST_TIMEOUT = 15  # seconds
REQUEST_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.0
SCRAPING_DELAY = 1.0  # seconds between consecutive requests to the same domain
MAX_CONCURRENT_WORKERS = 4

# HTTP Headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive"
}

# Image Validation Settings
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 150
MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_IMAGE_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
DISALLOWED_IMAGE_PATTERNS = [
    "1x1", "pixel", "tracking", "avatar", "logo-mobile", "logo-desktop",
    "favicon", "placeholder", "blank.gif", "spacer.gif", "advertisement",
    "ad-banner", "social-share", "icon-", "whatsapp", "instagram", "facebook"
]

# Headline Branding Suffixes to Clean
BRANDING_SUFFIXES = [
    " - Kompas.com",
    " - Kompas",
    " | Tempo",
    " - Tempo.co",
    " - CNN Indonesia",
    " - DetikNews",
    " - detikNews",
    " - Detik.com",
    " - detikcom",
    " - Liputan6.com",
    " - ANTARA News",
    " - Antara News",
    " - Tirto.ID",
    " - Tirto.id",
    " - BBC News Indonesia",
    " - BBC News",
    " - TurnBackHoax.ID",
    " - TurnBackHoax",
    " | Cek Fakta"
]

# Fact-Checking Verdict Keywords Mapping
FACT_CHECK_STATUS_MAP = {
    "FALSE": ["SALAH", "HOAKS", "HOAX", "PALSU", "FABRICATED", "TIDAK BENAR", "TIDAK SESUAI FAKTA", "BUKAN"],
    "MISLEADING": ["KLARIFIKASI", "DISINFORMASI", "MISLEADING", "MENYESATKAN", "KONTEN MANIPULASI", "MANIPULASI", "KONTEKS KELIRU", "PARODI"],
    "PARTIALLY_TRUE": ["SEBAGIAN BENAR", "KURANG TEPAT", "PERLU DILURUSKAN", "BELUM PASTI"],
    "TRUE": ["BENAR", "FAKTA", "SESUAI FAKTA", "VALID", "BUKAN HOAKS"],
    "UNVERIFIABLE": ["BELUM TERBUKTI", "TIDAK DAPAT DIBUKTIKAN", "TIDAK ADA BUKTI", "KLAIM TANPA DASAR"]
}

# Source Registry
SOURCES_CONFIG = {
    "kompas": {
        "name": "Kompas",
        "fact_check_url": "https://www.kompas.com/cekfakta",
        "news_url": "https://www.kompas.com/",
        "has_dedicated_fact_check": True
    },
    "tempo": {
        "name": "Tempo",
        "rss_url": "https://rss.tempo.co/nasional",
        "fact_check_rss": "https://rss.tempo.co/",
        "has_dedicated_fact_check": True
    },
    "detik": {
        "name": "Detik",
        "fact_check_url": "https://www.detik.com/tag/cek-fakta",
        "news_url": "https://news.detik.com/",
        "has_dedicated_fact_check": True
    },
    "cnn": {
        "name": "CNN Indonesia",
        "news_url": "https://www.cnnindonesia.com/nasional",
        "has_dedicated_fact_check": False
    },
    "antara": {
        "name": "Antara News",
        "fact_check_url": "https://www.antaranews.com/tag/hoaks",
        "news_rss": "https://www.antaranews.com/rss/terkini.xml",
        "has_dedicated_fact_check": True
    },
    "turnbackhoax": {
        "name": "TurnBackHoax (MAFINDO)",
        "rss_url": "https://turnbackhoax.id/feed/",
        "site_url": "https://turnbackhoax.id/",
        "has_dedicated_fact_check": True
    },
    "liputan6": {
        "name": "Liputan6",
        "fact_check_url": "https://www.liputan6.com/cek-fakta",
        "news_url": "https://www.liputan6.com/",
        "has_dedicated_fact_check": True
    },
    "tirto": {
        "name": "Tirto.id",
        "fact_check_url": "https://tirto.id/news/periksa-fakta",
        "news_url": "https://tirto.id/q/latest-news-D5ie",
        "has_dedicated_fact_check": True
    },
    "bbc": {
        "name": "BBC Indonesia",
        "news_url": "https://www.bbc.com/indonesia",
        "has_dedicated_fact_check": False
    }
}

