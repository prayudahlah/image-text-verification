"""
Deduplicator Module
Identifies duplicate articles using:
1. Canonical URL normalization (stripping tracking query parameters and fragments).
2. Exact & Normalized Headline Matching.
3. Headline Lexical Similarity (token set Jaccard distance with stricter threshold).
4. Caption/Content Similarity (detects same event covered with different headlines).
"""

import re
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Set, Tuple, List, Optional


class Deduplicator:
    TRACKING_PARAMS = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "page", "ref", "source", "outputType", "_gl"
    }

    def __init__(self, headline_threshold: float = 0.65, caption_threshold: float = 0.75):
        self.headline_threshold = headline_threshold
        self.caption_threshold = caption_threshold
        self.seen_urls: Set[str] = set()
        self.seen_thumbnails: Set[str] = set()
        self.seen_normalized_headlines: Set[str] = set()
        self.seen_headlines: List[Tuple[str, Set[str]]] = []
        self.seen_captions: List[Tuple[str, Set[str]]] = []

    def normalize_url(self, url: str) -> str:
        """Strips tracking query parameters and canonicalizes the URL."""
        if not url:
            return ""
        parsed = urlparse(url.strip())
        
        # Keep non-tracking query parameters
        filtered_queries = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if k.lower() not in self.TRACKING_PARAMS
        ]
        
        normalized_query = urlencode(filtered_queries)
        normalized_path = parsed.path.rstrip("/")
        
        # Reconstruct canonical URL (strip fragment, lowercase scheme and netloc)
        canonical = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            "",
            normalized_query,
            ""
        ))
        return canonical

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes punctuation and extra whitespaces for exact-match deduplication."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """Converts text into normalized word tokens."""
        if not text:
            return set()
        words = re.findall(r"\w+", text.lower())
        # Filter very short stopwords/single chars
        return {w for w in words if len(w) > 2}

    @staticmethod
    def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Calculates Jaccard similarity between two token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, url: str, headline: str, caption: Optional[str] = None, thumbnail_url: Optional[str] = None) -> Tuple[bool, str]:
        """
        Checks if the URL, headline, thumbnail, or article content/caption has already been seen.
        Returns (is_duplicate: bool, reason: str).
        """
        # 1. Check Canonical URL
        canonical_url = self.normalize_url(url)
        if canonical_url in self.seen_urls:
            return True, f"Canonical URL already scraped: {canonical_url}"

        # 2. Check Thumbnail URL
        if thumbnail_url:
            canonical_thumb = self.normalize_url(thumbnail_url)
            if canonical_thumb in self.seen_thumbnails:
                return True, f"Duplicate thumbnail image URL: {canonical_thumb}"

        # 3. Check Exact Normalized Headline
        clean_hl = self.clean_text(headline)
        if clean_hl and clean_hl in self.seen_normalized_headlines:
            return True, f"Exact identical headline: '{headline}'"

        # 4. Check Headline Semantic/Lexical Similarity
        hl_tokens = self.tokenize(headline)
        if hl_tokens:
            for existing_hl, existing_tokens in self.seen_headlines:
                sim = self.jaccard_similarity(hl_tokens, existing_tokens)
                if sim >= self.headline_threshold:
                    return True, f"Headline similarity ({sim:.2f} >= {self.headline_threshold}): '{headline}' vs '{existing_hl}'"

        # 5. Check Caption/Event Content Similarity
        if caption:
            caption_tokens = self.tokenize(caption)
            if len(caption_tokens) >= 5:
                for existing_cap, existing_c_tokens in self.seen_captions:
                    sim_cap = self.jaccard_similarity(caption_tokens, existing_c_tokens)
                    if sim_cap >= self.caption_threshold:
                        return True, f"News content/caption similarity ({sim_cap:.2f} >= {self.caption_threshold})"

        return False, ""

    def register(self, url: str, headline: str, caption: Optional[str] = None, thumbnail_url: Optional[str] = None):
        """Registers an article URL, thumbnail, headline, and caption into the deduplication index."""
        canonical_url = self.normalize_url(url)
        if canonical_url:
            self.seen_urls.add(canonical_url)

        if thumbnail_url:
            canonical_thumb = self.normalize_url(thumbnail_url)
            if canonical_thumb:
                self.seen_thumbnails.add(canonical_thumb)

        clean_hl = self.clean_text(headline)
        if clean_hl:
            self.seen_normalized_headlines.add(clean_hl)

        if headline:
            self.seen_headlines.append((headline, self.tokenize(headline)))

        if caption:
            c_tokens = self.tokenize(caption)
            if len(c_tokens) >= 5:
                self.seen_captions.append((caption, c_tokens))


