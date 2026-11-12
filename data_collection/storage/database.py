"""
SQLite Database Storage Module
Stores extracted Indonesian news articles and fact-checks with transactional safety.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import DB_PATH

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes the database schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thumbnail_url TEXT NOT NULL,
            headline TEXT NOT NULL,
            main_caption TEXT NOT NULL,
            source_name TEXT NOT NULL,
            article_url TEXT UNIQUE NOT NULL,
            publication_date TEXT,
            verification_type TEXT NOT NULL,
            fact_check_status TEXT NOT NULL,
            fact_check_source TEXT,
            claim_checked TEXT,
            verification_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_article_url ON articles(article_url);
        CREATE INDEX IF NOT EXISTS idx_verification_type ON articles(verification_type);
        CREATE INDEX IF NOT EXISTS idx_publication_date ON articles(publication_date);
        """
        with self.get_connection() as conn:
            conn.executescript(create_table_sql)
            conn.commit()

    def article_exists(self, article_url: str) -> bool:
        """Checks if an article with the given URL is already stored."""
        query = "SELECT 1 FROM articles WHERE article_url = ? LIMIT 1"
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (article_url,))
            return cur.fetchone() is not None

    def save_article(self, article: Dict[str, Any]) -> Optional[int]:
        """
        Inserts or updates an article record.
        Saves articles with valid verification status.
        Returns the article ID or None on failure/rejection.
        """
        if not article.get("fact_check_status") or article.get("fact_check_status") == "NOT_CHECKED":
            logger.debug(f"Rejected unverified article (status NOT_CHECKED): {article.get('article_url')}")
            return None
        sql = """
        INSERT INTO articles (
            thumbnail_url, headline, main_caption, source_name, article_url,
            publication_date, verification_type, fact_check_status,
            fact_check_source, claim_checked, verification_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_url) DO UPDATE SET
            thumbnail_url = excluded.thumbnail_url,
            headline = excluded.headline,
            main_caption = excluded.main_caption,
            publication_date = excluded.publication_date,
            verification_type = excluded.verification_type,
            fact_check_status = excluded.fact_check_status,
            fact_check_source = excluded.fact_check_source,
            claim_checked = excluded.claim_checked,
            verification_url = excluded.verification_url
        """
        params = (
            article.get("thumbnail_url"),
            article.get("headline"),
            article.get("main_caption"),
            article.get("source_name"),
            article.get("article_url"),
            article.get("publication_date"),
            article.get("verification_type"),
            article.get("fact_check_status"),
            article.get("fact_check_source"),
            article.get("claim_checked"),
            article.get("verification_url")
        )

        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                conn.commit()
                return cur.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Database error saving article {article.get('article_url')}: {e}")
            return None

    def get_articles(
        self,
        source: Optional[str] = None,
        mode: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries stored articles matching specific filter criteria.
        """
        conditions = []
        params = []

        if source and source.lower() != "all":
            conditions.append("LOWER(source_name) LIKE ?")
            params.append(f"%{source.lower()}%")

        if mode == "fact-check":
            conditions.append("verification_type = 'DEDICATED_FACT_CHECK'")

        if from_date:
            conditions.append("publication_date >= ?")
            params.append(from_date)

        if to_date:
            conditions.append("publication_date <= ?")
            params.append(to_date)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f" LIMIT {int(limit)}" if limit else ""

        sql = f"""
        SELECT 
            id, thumbnail_url, headline, main_caption, source_name, article_url,
            publication_date, verification_type, fact_check_status,
            fact_check_source, claim_checked, verification_url
        FROM articles
        {where_clause}
        ORDER BY id DESC
        {limit_clause}
        """

        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def count_articles(self) -> int:
        """Returns the total number of articles stored."""
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM articles")
            return cur.fetchone()[0]

