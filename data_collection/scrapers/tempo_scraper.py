"""
Tempo News Scraper
Scrapes articles from Tempo using official RSS feeds:
- https://rss.tempo.co/nasional
- https://rss.tempo.co/

Note: Tempo web pages employ strict Cloudflare WAF protection returning HTTP 403 on direct HTML requests.
Per project requirements, we gracefully utilize Tempo's publicly available RSS syndication feed which contains
full headline, high-resolution thumbnail image, 2-4 sentence caption, and publication date.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class TempoScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["tempo"]
        super().__init__(
            source_name=cfg["name"],
            base_url="https://www.tempo.co/"
        )
        self.rss_url = cfg["rss_url"]
        self.cached_entries: Dict[str, Any] = {}

    def discover_articles(self, mode: str = "all", limit: int = 10) -> List[str]:
        logger.info(f"Discovering articles from Tempo RSS: {self.rss_url}")
        entries = self.fetch_rss_feed(self.rss_url)
        article_urls: List[str] = []

        for entry in entries:
            link = entry.get("link")
            if not link:
                continue
            title = entry.get("title", "")
            
            # Cache feed entry for fallback extraction
            self.cached_entries[link] = entry

            if mode == "fact-check":
                # Check if it mentions hoaks / cek fakta / klarifikasi
                if any(kw in title.lower() for kw in ["cek fakta", "hoaks", "klarifikasi", "palsu", "salah"]):
                    article_urls.append(link)
            else:
                article_urls.append(link)

            if len(article_urls) >= limit:
                break

        return article_urls[:limit]

    def scrape_article(self, url: str, save_thumbnails: bool = False) -> Optional[Dict[str, Any]]:
        # 1. Try standard HTML scrape
        result = super().scrape_article(url, save_thumbnails=save_thumbnails)
        if result:
            return result

        # 2. If direct HTML was blocked (403), extract from cached RSS entry
        entry = self.cached_entries.get(url)
        if not entry:
            return None

        headline = entry.get("title", "").strip()
        headline = self.extractor.clean_headline(headline)
        caption = entry.get("summary", "").strip()
        thumb_url = entry.get("img")

        # Validate thumbnail
        if not thumb_url:
            return None

        is_valid, reason, _ = self.extractor.image_validator.validate_and_save(
            thumb_url, save_locally=save_thumbnails
        )
        if not is_valid:
            logger.warning(f"Rejected Tempo thumbnail ({thumb_url}): {reason}")
            return None

        # Parse publication date
        pub_date = None
        if entry.get("published_parsed"):
            try:
                tt = entry.published_parsed
                pub_date = f"{tt.tm_year:04d}-{tt.tm_mon:02d}-{tt.tm_mday:02d}"
            except Exception:
                pass

        # Fact check validation
        validation = self.extractor.factcheck_validator.validate(
            headline=headline,
            article_url=url,
            source_name=self.source_name,
            body_text=caption
        )

        if validation["verification_type"] != "DEDICATED_FACT_CHECK" or validation["fact_check_status"] == "NOT_CHECKED":
            return None

        return {
            "thumbnail_url": thumb_url,
            "headline": headline,
            "main_caption": caption,
            "source_name": self.source_name,
            "article_url": url,
            "publication_date": pub_date,
            "verification_type": validation["verification_type"],
            "fact_check_status": validation["fact_check_status"],
            "fact_check_source": validation["fact_check_source"],
            "claim_checked": validation["claim_checked"],
            "verification_url": validation["verification_url"]
        }

