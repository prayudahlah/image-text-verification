"""
Kompas News & Cek Fakta Scraper
Scrapes articles from:
- https://www.kompas.com/cekfakta (Dedicated Fact-Checks)
- https://www.kompas.com/tag/cek-fakta?page={page} (Paginated Fact-Checks)
"""

import logging
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class KompasScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["kompas"]
        super().__init__(
            source_name=cfg["name"],
            base_url=cfg["news_url"]
        )
        self.fact_check_url = cfg["fact_check_url"]
        self.tag_url = "https://www.kompas.com/tag/cek-fakta"

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []
        page = 1
        max_pages = max(100, (limit // 15) + 10)
        consecutive_empty = 0

        while len(article_urls) < limit and page <= max_pages:
            page_url = f"https://www.kompas.com/tag/berita?page={page}"
            logger.info(f"Paginating Kompas tag/berita: {page_url}")
            tag_html = self.fetch_html(page_url)
            if not tag_html:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                page += 1
                continue

            soup = BeautifulSoup(tag_html, "html.parser")
            page_articles = 0
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0]
                if "/read/" in href and "kompas.com" in href:
                    page_articles += 1
                    if href not in article_urls:
                        article_urls.append(href)
                        if len(article_urls) >= limit:
                            return article_urls[:limit]

            if page_articles == 0:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
            else:
                consecutive_empty = 0

            page += 1

        return article_urls[:limit]

    def scrape_article(self, url: str, save_thumbnails: bool = False) -> Optional[Dict[str, Any]]:
        return super().scrape_article(url, save_thumbnails=save_thumbnails)
