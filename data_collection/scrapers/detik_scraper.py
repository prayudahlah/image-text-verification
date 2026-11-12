"""
Detik News & Cek Fakta Scraper
Scrapes articles from:
- https://www.detik.com/tag/cek-fakta?sortby=time&page={page} (Dedicated Fact-Checks)
"""

import logging
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class DetikScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["detik"]
        super().__init__(
            source_name=cfg["name"],
            base_url="https://www.detik.com/"
        )
        self.fact_check_base = cfg["fact_check_url"]

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []

        page = 1
        max_pages = max(20, (limit // 10) + 5)
        consecutive_empty = 0

        while len(article_urls) < limit and page <= max_pages:
            p_url = f"{self.fact_check_base}?sortby=time&page={page}"
            logger.info(f"Paginating Detik Cek Fakta: {p_url}")
            html = self.fetch_html(p_url)
            if not html:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                page += 1
                continue

            soup = BeautifulSoup(html, "html.parser")
            page_articles = 0
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0]
                if "detik.com" in href and ("/berita/" in href or "/read/" in href or "/d-" in href):
                    page_articles += 1
                    if href not in article_urls:
                        article_urls.append(href)
                        if len(article_urls) >= limit:
                            return article_urls[:limit]

            if page_articles == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
            else:
                consecutive_empty = 0

            page += 1

        return article_urls[:limit]

    def scrape_article(self, url: str, save_thumbnails: bool = False) -> Optional[Dict[str, Any]]:
        result = super().scrape_article(url, save_thumbnails=save_thumbnails)
        if not result:
            return None
        # Strict fact check enforcement
        if result.get("verification_type") != "DEDICATED_FACT_CHECK" or result.get("fact_check_status") == "NOT_CHECKED":
            return None
        return result
