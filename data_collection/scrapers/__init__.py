"""
Scrapers Package Initialization
Registers and exports all source-specific news scrapers.
"""

from scrapers.base_scraper import BaseScraper
from scrapers.kompas_scraper import KompasScraper
from scrapers.tempo_scraper import TempoScraper
from scrapers.detik_scraper import DetikScraper
from scrapers.cnn_scraper import CnnScraper
from scrapers.antara_scraper import AntaraScraper
from scrapers.factcheck_scraper import (
    TurnBackHoaxScraper,
    Liputan6Scraper,
    TirtoScraper,
    BbcScraper
)

SCRAPER_REGISTRY = {
    "kompas": KompasScraper,
    "tempo": TempoScraper,
    "detik": DetikScraper,
    "cnn": CnnScraper,
    "antara": AntaraScraper,
    "turnbackhoax": TurnBackHoaxScraper,
    "liputan6": Liputan6Scraper,
    "tirto": TirtoScraper,
    "bbc": BbcScraper
}

__all__ = [
    "BaseScraper",
    "KompasScraper",
    "TempoScraper",
    "DetikScraper",
    "CnnScraper",
    "AntaraScraper",
    "TurnBackHoaxScraper",
    "Liputan6Scraper",
    "TirtoScraper",
    "BbcScraper",
    "SCRAPER_REGISTRY"
]

