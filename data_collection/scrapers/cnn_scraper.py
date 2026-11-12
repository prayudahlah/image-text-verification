"""
CNN Indonesia News Scraper
Scrapes articles from:
- https://www.cnnindonesia.com/nasional
"""

import logging
from typing import List
from bs4 import BeautifulSoup
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class CnnScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["cnn"]
        super().__init__(
            source_name=cfg["name"],
            base_url="https://www.cnnindonesia.com/"
        )
        self.news_url = cfg["news_url"]

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []
        page = 1
        max_pages = max(100, (limit // 15) + 10)

        while len(article_urls) < limit and page <= max_pages:
            page_url = f"https://www.cnnindonesia.com/indeks?page={page}"
            logger.info(f"Paginating CNN Indonesia Indeks: {page_url}")
            html = self.fetch_html(page_url)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            found = 0
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0]
                if "cnnindonesia.com/" in href and any(c.isdigit() for c in href) and "/foto/" not in href and "/video/" not in href:
                    found += 1
                    if href not in article_urls:
                        article_urls.append(href)
                        if len(article_urls) >= limit:
                            return article_urls[:limit]

            if found == 0:
                break
            page += 1

        return article_urls[:limit]

