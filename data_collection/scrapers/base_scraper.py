"""
Base Scraper Module
Provides shared scraping infrastructure:
- Request retry mechanism with exponential backoff
- Rate limiting and politeness delays
- robots.txt compliance checking
- RSS and HTML fetch utilities
- Error handling and logging
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import feedparser

from config import (
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    REQUEST_RETRIES,
    RETRY_BACKOFF_FACTOR,
    SCRAPING_DELAY
)
from processors.article_extractor import ArticleExtractor
from processors.image_validator import ImageValidator
from processors.factcheck_validator import FactCheckValidator

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    def __init__(
        self,
        source_name: str,
        base_url: str,
        extractor: Optional[ArticleExtractor] = None,
        respect_robots_txt: bool = True
    ):
        self.source_name = source_name
        self.base_url = base_url
        self.respect_robots_txt = respect_robots_txt
        self.session = self._create_session()
        self.extractor = extractor or ArticleExtractor(
            image_validator=ImageValidator(session=self.session),
            factcheck_validator=FactCheckValidator()
        )
        self.last_request_time = 0.0
        self.robot_parser: Optional[RobotFileParser] = None
        if self.respect_robots_txt:
            self._load_robots_txt()

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        
        # Configure robust retries for transient errors
        retry_strategy = Retry(
            total=REQUEST_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _load_robots_txt(self):
        try:
            parsed = urlparse(self.base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            # Fetch with custom headers to prevent 403 on robots.txt
            resp = self.session.get(robots_url, timeout=5)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                self.robot_parser = rp
                logger.debug(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.debug(f"Could not load robots.txt for {self.base_url}: {e}")

    def is_allowed_by_robots(self, url: str) -> bool:
        if not self.robot_parser:
            return True
        user_agent = DEFAULT_HEADERS.get("User-Agent", "*")
        try:
            return self.robot_parser.can_fetch(user_agent, url)
        except Exception:
            return True

    def _polite_delay(self):
        """Enforces a courteous scraping delay between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < SCRAPING_DELAY:
            time.sleep(SCRAPING_DELAY - elapsed)
        self.last_request_time = time.time()

    def fetch_html(self, url: str) -> Optional[str]:
        """
        Safely fetches HTML content from a URL with retry, delay, and error logging.
        """
        if self.respect_robots_txt and not self.is_allowed_by_robots(url):
            logger.warning(f"robots.txt disallows scraping {url}")
            return None

        self._polite_delay()
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 403:
                logger.warning(f"HTTP 403 Forbidden for {url} (Site employs anti-bot protection)")
                return None
            elif resp.status_code == 404:
                logger.warning(f"HTTP 404 Not Found: {url}")
                return None
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout requesting {url} after {REQUEST_TIMEOUT}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error requesting {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def fetch_rss_feed(self, rss_url: str) -> List[Dict[str, Any]]:
        """Parses an RSS feed and returns entries."""
        self._polite_delay()
        try:
            resp = self.session.get(rss_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch RSS feed {rss_url}: status {resp.status_code}")
                return []
            feed = feedparser.parse(resp.text)
            return feed.entries
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            return []

    def fetch_with_playwright(self, url: str) -> Optional[str]:
        """
        Optional fallback using Playwright for JavaScript-rendered sites.
        """
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    user_agent=DEFAULT_HEADERS["User-Agent"]
                )
                page.goto(url, wait_until="networkidle", timeout=15000)
                html = page.content()
                browser.close()
                return html
        except Exception as e:
            logger.warning(f"Playwright rendering failed for {url}: {e}")
            return None

    @abstractmethod
    def discover_articles(self, mode: str = "all", limit: int = 10) -> List[str]:
        """Discovers article URLs according to mode ('all' or 'fact-check')."""
        pass

    def scrape_article(self, url: str, save_thumbnails: bool = False) -> Optional[Dict[str, Any]]:
        """Scrapes an individual article from its URL."""
        html = self.fetch_html(url)
        if not html:
            return None
        return self.extractor.extract_from_html(
            html_content=html,
            article_url=url,
            source_name=self.source_name,
            save_thumbnails=save_thumbnails
        )

