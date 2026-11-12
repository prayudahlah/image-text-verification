"""
Article Extractor Module
Extracts the three primary content elements required:
1. Thumbnail Image (validated with Pillow)
2. News Headline (with website branding suffixes removed)
3. Main News Caption (concise 2-4 sentences from meta or lead paragraphs)

Also extracts publication date and integrates with factcheck validation.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from config import BRANDING_SUFFIXES
from processors.image_validator import ImageValidator
from processors.factcheck_validator import FactCheckValidator

logger = logging.getLogger(__name__)


class ArticleExtractor:
    def __init__(self, image_validator: Optional[ImageValidator] = None, factcheck_validator: Optional[FactCheckValidator] = None):
        self.image_validator = image_validator or ImageValidator()
        self.factcheck_validator = factcheck_validator or FactCheckValidator()

    @staticmethod
    def clean_headline(raw_headline: str) -> str:
        """Removes website branding suffixes from the headline while preserving the news text."""
        if not raw_headline:
            return ""
        
        headline = raw_headline.strip()
        # Clean standard suffixes
        for suffix in BRANDING_SUFFIXES:
            if headline.endswith(suffix):
                headline = headline[:-len(suffix)].strip()
            elif suffix.lower() in headline.lower():
                # Check case-insensitive suffix
                idx = headline.lower().rfind(suffix.lower())
                if idx > 10 and idx + len(suffix) >= len(headline) - 3:
                    headline = headline[:idx].strip()

        # Remove trailing pipes or hyphens
        headline = re.sub(r"\s*[\-\|]\s*$", "", headline).strip()
        return headline

    @staticmethod
    def extract_headline(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts news headline prioritizing og:title, twitter:title, then h1.
        Returns (cleaned_headline, raw_headline).
        """
        raw_headline = None

        # 1. Try og:title
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "og:title"})
        if og_title and og_title.get("content"):
            raw_headline = og_title["content"].strip()

        # 2. Try twitter:title
        if not raw_headline:
            tw_title = soup.find("meta", attrs={"name": "twitter:title"}) or soup.find("meta", property="twitter:title")
            if tw_title and tw_title.get("content"):
                raw_headline = tw_title["content"].strip()

        # 3. Try h1
        if not raw_headline:
            h1 = soup.find("h1")
            if h1:
                raw_headline = h1.get_text(separator=" ", strip=True)

        # 4. Fallback to <title> tag
        if not raw_headline:
            title_tag = soup.find("title")
            if title_tag:
                raw_headline = title_tag.get_text(strip=True)

        if not raw_headline:
            return None, None

        cleaned = ArticleExtractor.clean_headline(raw_headline)
        return cleaned, raw_headline

    @staticmethod
    def generate_concise_caption(body_text: str, max_sentences: int = 3) -> Optional[str]:
        """Generates a concise 2-4 sentence summary using the article's actual content."""
        if not body_text:
            return None

        # Split text into sentences
        # Handle Indonesian punctuation
        sentences = re.split(r"(?<=[.!?])\s+", body_text.strip())
        meaningful_sentences = []
        for s in sentences:
            s_clean = s.strip()
            # Ignore short boilerplate, dates, or navigation strings
            if len(s_clean) > 25 and not any(kw in s_clean.lower() for kw in ["baca juga", "simak juga", "halaman selanjutnya", "advertisement"]):
                meaningful_sentences.append(s_clean)
                if len(meaningful_sentences) >= max_sentences:
                    break

        if meaningful_sentences:
            caption = " ".join(meaningful_sentences)
            return caption

        return None

    @staticmethod
    def extract_caption(soup: BeautifulSoup, body_text: str = "") -> Optional[str]:
        """
        Extracts news caption prioritizing og:description, meta description, then opening lead.
        Ensures caption is concise (2-4 sentences).
        """
        caption_candidate = None

        # 1. Try og:description
        og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "og:description"})
        if og_desc and og_desc.get("content"):
            caption_candidate = og_desc["content"].strip()

        # 2. Try meta description
        if not caption_candidate:
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                caption_candidate = meta_desc["content"].strip()

        # 3. Try twitter:description
        if not caption_candidate:
            tw_desc = soup.find("meta", attrs={"name": "twitter:description"})
            if tw_desc and tw_desc.get("content"):
                caption_candidate = tw_desc["content"].strip()

        # If meta description exists and is substantive (at least 20 chars), check if it needs sentence capping
        if caption_candidate and len(caption_candidate) > 20:
            # Clean up whitespace
            caption_candidate = re.sub(r"\s+", " ", caption_candidate)
            return caption_candidate

        # 4. Fallback: Generate concise 2-4 sentence lead from body
        return ArticleExtractor.generate_concise_caption(body_text, max_sentences=3)

    def extract_thumbnail_url(self, soup: BeautifulSoup, base_url: str, save_locally: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts thumbnail URL, validating candidates using Pillow.
        Returns (valid_thumbnail_url, local_file_path).
        """
        candidates: List[str] = []

        # 1. og:image
        og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
        if og_img and og_img.get("content"):
            candidates.append(og_img["content"].strip())

        # 2. twitter:image
        tw_img = soup.find("meta", attrs={"name": "twitter:image"}) or soup.find("meta", property="twitter:image")
        if tw_img and tw_img.get("content"):
            candidates.append(tw_img["content"].strip())

        # 3. JSON-LD image
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    img = data.get("image")
                    if isinstance(img, str):
                        candidates.append(img)
                    elif isinstance(img, dict) and img.get("url"):
                        candidates.append(img["url"])
                    elif isinstance(img, list) and img:
                        candidates.append(img[0] if isinstance(img[0], str) else img[0].get("url", ""))
            except Exception:
                continue

        # 4. Main article figure / img tags
        for figure in soup.find_all(["figure", "div"], class_=re.compile(r"(photo|image|thumb|featured|lead)", re.I)):
            img = figure.find("img")
            if img:
                src = img.get("src") or img.get("data-src") or img.get("data-original")
                if src:
                    candidates.append(src.strip())

        # 5. Any substantive img tag in article
        article_elem = soup.find(["article", "main"]) or soup
        for img in article_elem.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if src:
                candidates.append(src.strip())

        # Validate candidates in order
        seen = set()
        for cand in candidates:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)

            abs_url = urljoin(base_url, cand)
            is_valid, reason, local_path = self.image_validator.validate_and_save(abs_url, save_locally=save_locally)
            if is_valid:
                return abs_url, local_path
            else:
                logger.debug(f"Candidate image rejected ({abs_url}): {reason}")

        return None, None

    @staticmethod
    def extract_body_text(soup: BeautifulSoup) -> str:
        """Extracts cleaned textual paragraphs from the article body."""
        # Find main article container
        container = (
            soup.find("div", class_=re.compile(r"(read__content|detail__body-text|detail-text|content-detail|post-content|article-content|read__article|content-article|article__content)", re.I))
            or soup.find("article")
            or soup.find("main")
            or soup.body
        )

        if not container:
            return ""

        paragraphs = []
        for p in container.find_all(["p", "div"]):
            # Skip hidden elements or disclaimers
            if p.name == "div" and p.find_all("p"):
                continue  # Avoid duplicate nested text
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 30 and not any(kw in text.lower() for kw in ["baca juga", "simak juga", "halaman selanjutnya", "pilihan editor"]):
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    @staticmethod
    def extract_publication_date(soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extracts publication date formatted as YYYY-MM-DD."""
        # 1. Meta article:published_time
        meta_date = (
            soup.find("meta", property="article:published_time")
            or soup.find("meta", attrs={"name": "pubdate"})
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("meta", property="og:updated_time")
        )
        if meta_date and meta_date.get("content"):
            content = meta_date["content"]
            m = re.search(r"(\d{4}-\d{2}-\d{2})", content)
            if m:
                return m.group(1)

        # 2. JSON-LD datePublished
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    date_val = data.get("datePublished") or data.get("dateModified")
                    if date_val:
                        m = re.search(r"(\d{4}-\d{2}-\d{2})", str(date_val))
                        if m:
                            return m.group(1)
            except Exception:
                continue

        # 3. Date pattern in URL: /2026/11/10/ or /20261110/
        url_match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if url_match:
            return f"{url_match.group(1)}-{url_match.group(2)}-{url_match.group(3)}"

        url_match2 = re.search(r"/(\d{4})(\d{2})(\d{2})", url)
        if url_match2:
            return f"{url_match2.group(1)}-{url_match2.group(2)}-{url_match2.group(3)}"

        return None

    def extract_from_html(
        self,
        html_content: str,
        article_url: str,
        source_name: str,
        save_thumbnails: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts all required fields from raw HTML string.
        Returns a dictionary or None if mandatory elements (headline, thumbnail) are missing.
        """
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Headline
        headline, raw_headline = self.extract_headline(soup)
        if not headline:
            logger.warning(f"Rejected article {article_url}: Missing or empty headline.")
            return None

        # 2. Body Text & Caption
        body_text = self.extract_body_text(soup)
        caption = self.extract_caption(soup, body_text)
        if not caption:
            # Fallback to headline if no caption available
            caption = headline

        # 3. Thumbnail
        thumbnail_url, _ = self.extract_thumbnail_url(soup, article_url, save_locally=save_thumbnails)
        if not thumbnail_url:
            logger.warning(f"Rejected article {article_url}: No valid thumbnail image found meeting quality criteria.")
            return None

        # 4. Publication Date
        pub_date = self.extract_publication_date(soup, article_url)

        # 5. Fact Check Validation
        validation = self.factcheck_validator.validate(
            headline=headline,
            article_url=article_url,
            source_name=source_name,
            body_text=body_text,
            raw_headline=raw_headline
        )

        return {
            "thumbnail_url": thumbnail_url,
            "headline": headline,
            "main_caption": caption,
            "source_name": source_name,
            "article_url": article_url,
            "publication_date": pub_date,
            "verification_type": validation["verification_type"],
            "fact_check_status": validation["fact_check_status"],
            "fact_check_source": validation["fact_check_source"],
            "claim_checked": validation["claim_checked"],
            "verification_url": validation["verification_url"]
        }
