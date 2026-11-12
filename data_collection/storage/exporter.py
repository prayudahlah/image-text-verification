"""
Data Exporter Module
Exports structured news articles and fact-checks into:
1. JSON file (strictly following project schema)
2. CSV file (UTF-8 encoded)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from config import JSON_OUTPUT_PATH, CSV_OUTPUT_PATH

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(self, json_path: Optional[Path] = None, csv_path: Optional[Path] = None):
        self.json_path = Path(json_path or JSON_OUTPUT_PATH)
        self.csv_path = Path(csv_path or CSV_OUTPUT_PATH)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def export_to_json(self, articles: List[Dict[str, Any]]) -> str:
        """
        Exports articles to JSON file strictly adhering to schema:
        {
          "articles": [ ... ]
        }
        Missing values remain null, never fabricated.
        """
        payload = {
            "articles": [
                {
                    "id": item.get("id"),
                    "thumbnail_url": item.get("thumbnail_url"),
                    "headline": item.get("headline"),
                    "main_caption": item.get("main_caption"),
                    "source_name": item.get("source_name"),
                    "article_url": item.get("article_url"),
                    "publication_date": item.get("publication_date"),
                    "verification_type": item.get("verification_type"),
                    "fact_check_status": item.get("fact_check_status"),
                    "fact_check_source": item.get("fact_check_source"),
                    "claim_checked": item.get("claim_checked")
                }
                for item in articles
            ]
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully exported {len(articles)} articles to {self.json_path}")
        return str(self.json_path)

    def export_to_csv(self, articles: List[Dict[str, Any]]) -> str:
        """
        Exports articles to CSV file using pandas with UTF-8 encoding.
        """
        columns = [
            "id", "thumbnail_url", "headline", "main_caption", "source_name",
            "article_url", "publication_date", "verification_type",
            "fact_check_status", "fact_check_source", "claim_checked"
        ]

        if not articles:
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(articles)
            # Reorder columns according to schema
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Successfully exported {len(articles)} articles to {self.csv_path}")
        return str(self.csv_path)

    def export_all(self, articles: List[Dict[str, Any]]):
        """Exports to both JSON and CSV files."""
        self.export_to_json(articles)
        self.export_to_csv(articles)

