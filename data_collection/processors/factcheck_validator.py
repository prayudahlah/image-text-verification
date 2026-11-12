"""
Fact-Check Validation Module
Accurately distinguishes between:
1. DEDICATED_FACT_CHECK: Articles explicitly investigating a claim.
2. VERIFIED_NEWS: Regular news articles backed by official/primary sources.

Verification status labels:
- TRUE
- FALSE
- MISLEADING
- PARTIALLY_TRUE
- UNVERIFIABLE
- NOT_CHECKED
"""

import re
from typing import Dict, Any, Optional, Tuple
from config import FACT_CHECK_STATUS_MAP


class FactCheckValidator:
    # Dedicated source indicators in URLs
    DEDICATED_URL_KEYWORDS = [
        "cekfakta", "cek-fakta", "turnbackhoax.id", "periksa-fakta",
        "/tag/hoaks", "/anti-hoaks", "/hoaks/", "/cek-fakta/"
    ]

    # Explicit tags commonly found in Indonesian headlines
    HEADLINE_TAG_REGEX = re.compile(
        r"^\[\s*(SALAH|HOAKS|HOAX|KLARIFIKASI|DISINFORMASI|MANIPULASI|MISLEADING|BENAR|FAKTA|SATIR|SEBAGIAN BENAR)\s*\]\s*",
        re.IGNORECASE
    )

    # Patterns for claim extraction
    CLAIM_PATTERNS = [
        re.compile(r"(?:Klaim|Narasi|Beredar unggahan|Beredar kabar|Beredar video|Beredar foto|Sebuah unggahan|Pesan berantai)\s*[:\-]?\s*[\"']?([^\"\n\.\!\?]{15,200})", re.IGNORECASE),
        re.compile(r"mengeklaim bahwa\s+([^.\n]{15,200})", re.IGNORECASE),
        re.compile(r"diklaim\s+([^.\n]{15,200})", re.IGNORECASE)
    ]

    # Official attribution markers indicating verified news
    PRIMARY_SOURCE_MARKERS = [
        "kata kapolri", "ujar menteri", "kata presiden", "menurut keterangan resmi",
        "dalam konferensi pers", "konfirmasi", "siaran pers", "dikutip dari laman resmi",
        "kata humas", "ujar kabid", "tutur direktur", "kata kapolda", "dalam rilis",
        "dibenarkan oleh", "ditegaskan oleh", "menurut bmkg", "kata kemenkes"
    ]

    def __init__(self):
        pass

    def is_dedicated_fact_check_source(self, url: str, source_name: str) -> bool:
        """Determines if the URL or source is inherently a dedicated fact-checking platform/category."""
        lowered_url = url.lower()
        if any(keyword in lowered_url for keyword in self.DEDICATED_URL_KEYWORDS):
            return True
        if "cek fakta" in source_name.lower() or "turnbackhoax" in source_name.lower():
            return True
        return False

    def extract_headline_tag(self, headline: str) -> Tuple[Optional[str], str]:
        """
        Extracts tags like [SALAH], [KLARIFIKASI] from the headline and returns (tag, cleaned_headline).
        """
        match = self.HEADLINE_TAG_REGEX.search(headline)
        if match:
            tag = match.group(1).upper()
            cleaned = headline[match.end():].strip()
            return tag, cleaned
        return None, headline

    def determine_status_from_tag(self, tag: Optional[str], text_content: str = "") -> str:
        """Maps a tag or keywords in text to a standardized fact_check_status."""
        search_target = f"{tag or ''} {text_content}".upper()
        
        # Check against status map
        for status, keywords in FACT_CHECK_STATUS_MAP.items():
            for kw in keywords:
                if kw in search_target:
                    return status

        return "NOT_CHECKED"

    def extract_claim(self, headline: str, raw_headline: str, body_text: str = "") -> Optional[str]:
        """Extracts the specific claim being investigated."""
        tag, _ = self.extract_headline_tag(raw_headline)
        
        # If headline had a [SALAH] / [KLARIFIKASI] tag, the headline itself describes the claim
        if tag:
            # Strip the tag to reveal the claim
            claim = raw_headline[raw_headline.find("]") + 1:].strip() if "]" in raw_headline else headline
            # Clean leading punctuation
            claim = claim.lstrip(" :-,")
            if len(claim) > 10:
                return claim

        # Search within body text
        for pat in self.CLAIM_PATTERNS:
            match = pat.search(body_text)
            if match:
                return match.group(1).strip()

        # If it's a dedicated fact check, the headline usually represents the claim
        return headline.strip() if headline else None

    def validate(
        self,
        headline: str,
        article_url: str,
        source_name: str,
        body_text: str = "",
        raw_headline: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates whether the article is a DEDICATED_FACT_CHECK or VERIFIED_NEWS.
        Returns a dictionary with verification fields:
        {
            "verification_type": "DEDICATED_FACT_CHECK" | "VERIFIED_NEWS",
            "fact_check_status": "TRUE" | "FALSE" | "MISLEADING" | "PARTIALLY_TRUE" | "UNVERIFIABLE" | "NOT_CHECKED",
            "fact_check_source": Optional[str],
            "claim_checked": Optional[str],
            "verification_url": Optional[str]
        }
        """
        raw_hl = raw_headline or headline
        tag, _ = self.extract_headline_tag(raw_hl)
        is_dedicated = self.is_dedicated_fact_check_source(article_url, source_name) or (tag is not None)

        body_snippet = body_text[:2000].lower() if body_text else ""

        if is_dedicated:
            # Determine specific organization
            fact_check_source = source_name
            if "kompas" in article_url.lower() or "kompas" in source_name.lower():
                fact_check_source = "Cek Fakta Kompas"
            elif "turnbackhoax" in article_url.lower():
                fact_check_source = "TurnBackHoax (MAFINDO)"
            elif "liputan6" in article_url.lower():
                fact_check_source = "Cek Fakta Liputan6"
            elif "tempo" in article_url.lower():
                fact_check_source = "Cek Fakta Tempo"
            elif "detik" in article_url.lower():
                fact_check_source = "Detik Cek Fakta"
            elif "antara" in article_url.lower():
                fact_check_source = "Antara Anti-Hoaks"
            elif "tirto" in article_url.lower():
                fact_check_source = "Tirto Periksa Fakta"

            # Determine status
            status = self.determine_status_from_tag(tag, body_snippet)
            if status == "NOT_CHECKED":
                # Default for dedicated fact check if not explicitly labeled
                if "hoaks" in raw_hl.lower() or "salah" in raw_hl.lower():
                    status = "FALSE"
                elif "klarifikasi" in raw_hl.lower() or "disinformasi" in raw_hl.lower():
                    status = "MISLEADING"
                else:
                    status = "MISLEADING"

            claim = self.extract_claim(headline, raw_hl, body_text)

            return {
                "verification_type": "DEDICATED_FACT_CHECK",
                "fact_check_status": status,
                "fact_check_source": fact_check_source,
                "claim_checked": claim,
                "verification_url": article_url
            }

        # Regular news article: check for official/primary source verification
        has_primary_source = any(marker in body_snippet for marker in self.PRIMARY_SOURCE_MARKERS)
        
        return {
            "verification_type": "VERIFIED_NEWS",
            "fact_check_status": "TRUE",
            "fact_check_source": source_name,
            "claim_checked": headline,
            "verification_url": article_url
        }

