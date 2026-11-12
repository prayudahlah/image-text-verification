"""
Antara News Scraper
Scrapes articles from:
- https://www.antaranews.com/tag/hoaks/{page} (Anti-Hoaks & Clarifications)
"""

import logging
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class AntaraScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["antara"]
        super().__init__(
            source_name=cfg["name"],
            base_url="https://www.antaranews.com/"
        )
        self.fact_check_base = "https://www.antaranews.com/tag/hoaks"

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []

        # 1. Fact-check / hoaks tag with pagination
        page = 1
        max_pages = max(50, (limit // 20) + 10)
        consecutive_empty = 0

        while len(article_urls) < limit and page <= max_pages:
            p_url = f"{self.fact_check_base}/{page}"
            logger.info(f"Paginating Antara tag hoaks: {p_url}")
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
                if "/berita/" in href and "antaranews.com" in href:
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
