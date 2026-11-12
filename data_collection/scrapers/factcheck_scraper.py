"""
Dedicated Fact-Checkers and Verified Portals Scraper Module
Implements paginated adapters for:
1. TurnBackHoax (MAFINDO) with multi-page crawling
2. Liputan6 Cek Fakta with multi-page crawling
3. Tirto.id Periksa Fakta
4. BBC News Indonesia
"""

import logging
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from scrapers.base_scraper import BaseScraper
from config import SOURCES_CONFIG

logger = logging.getLogger(__name__)


class TurnBackHoaxScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["turnbackhoax"]
        super().__init__(
            source_name=cfg["name"],
            base_url=cfg["site_url"]
        )
        self.rss_url = cfg["rss_url"]
        self.articles_url = "https://turnbackhoax.id/articles"

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []

        # 1. Fetch from RSS feed first
        logger.info(f"Discovering fact-checks from TurnBackHoax RSS: {self.rss_url}")
        entries = self.fetch_rss_feed(self.rss_url)
        for entry in entries:
            link = entry.get("link")
            if link and link not in article_urls:
                article_urls.append(link)
                if len(article_urls) >= limit:
                    return article_urls[:limit]

        # 2. Paginate through /articles?page=p
        page = 1
        max_pages = max(120, (limit // 8) + 10)
        consecutive_empty = 0

        while len(article_urls) < limit and page <= max_pages:
            p_url = f"{self.articles_url}?page={page}"
            logger.info(f"Paginating TurnBackHoax: {p_url}")
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
                if "/articles/" in href and href != "/articles":
                    abs_url = href if href.startswith("http") else f"https://turnbackhoax.id{href}"
                    page_articles += 1
                    if abs_url not in article_urls:
                        article_urls.append(abs_url)
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
        html = self.fetch_html(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")
        
        # Headline
        headline, raw_headline = self.extractor.extract_headline(soup)
        if not headline:
            return None

        # TurnBackHoax specific thumbnail extraction (located inside figure)
        thumb_url = None
        figure = soup.find("figure")
        if figure:
            img = figure.find("img")
            if img:
                thumb_url = img.get("src")

        # Fallback to standard extractor
        if not thumb_url:
            thumb_url, _ = self.extractor.extract_thumbnail_url(soup, url, save_locally=save_thumbnails)
        else:
            is_valid, reason, _ = self.extractor.image_validator.validate_and_save(thumb_url, save_locally=save_thumbnails)
            if not is_valid:
                logger.debug(f"TurnBackHoax figure image rejected: {reason}")
                thumb_url = None

        if not thumb_url:
            logger.warning(f"Rejected TurnBackHoax article {url}: No valid thumbnail image found.")
            return None

        # Body text & Caption
        body_text = self.extractor.extract_body_text(soup)
        caption = self.extractor.extract_caption(soup, body_text)
        if not caption:
            caption = headline

        pub_date = self.extractor.extract_publication_date(soup, url)

        validation = self.extractor.factcheck_validator.validate(
            headline=headline,
            article_url=url,
            source_name=self.source_name,
            body_text=body_text,
            raw_headline=raw_headline
        )

        # STRICT: Must be DEDICATED_FACT_CHECK
        if validation["verification_type"] != "DEDICATED_FACT_CHECK":
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


class Liputan6Scraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["liputan6"]
        super().__init__(
            source_name=cfg["name"],
            base_url=cfg["news_url"]
        )
        self.fact_check_url = cfg["fact_check_url"]
        self.tag_url = "https://www.liputan6.com/tag/cek-fakta"

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []
        page = 1
        max_pages = max(100, (limit // 20) + 10)
        consecutive_empty = 0

        while len(article_urls) < limit and page <= max_pages:
            page_url = f"https://www.liputan6.com/tag/berita-hari-ini?page={page}"
            logger.info(f"Paginating Liputan6 Tag Berita: {page_url}")
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
                if "/read/" in href and "liputan6.com" in href and "/cek-fakta/" not in href:
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


class TirtoScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["tirto"]
        super().__init__(
            source_name=cfg["name"],
            base_url="https://tirto.id/"
        )
        self.fact_check_url = cfg["fact_check_url"]
        self.news_url = cfg["news_url"]

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []
        target_url = self.fact_check_url
        logger.info(f"Discovering articles from Tirto: {target_url}")
        html = self.fetch_html(target_url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0]
                if "tirto.id/" in href and href.count("-") >= 3 and not href.startswith("https://tirto.id/q/"):
                    if href not in article_urls:
                        article_urls.append(href)
                        if len(article_urls) >= limit:
                            break
        return article_urls[:limit]


class BbcScraper(BaseScraper):
    def __init__(self):
        cfg = SOURCES_CONFIG["bbc"]
        super().__init__(
            source_name=cfg["name"],
            base_url=cfg["news_url"]
        )
        self.rss_url = "https://feeds.bbci.co.uk/indonesia/rss.xml"

    def discover_articles(self, mode: str = "all", limit: int = 100) -> List[str]:
        article_urls: List[str] = []
        entries = self.fetch_rss_feed(self.rss_url)
        for e in entries:
            link = e.get("link")
            if link and link not in article_urls:
                article_urls.append(link)

        html = self.fetch_html("https://www.bbc.com/indonesia")
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].split("?")[0]
                if ("/indonesia/articles/" in href or "/indonesia/indonesia-" in href):
                    full_url = href if href.startswith("http") else f"https://www.bbc.com{href}"
                    if full_url not in article_urls:
                        article_urls.append(full_url)

        return article_urls[:limit]
