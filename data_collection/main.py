"""
Indonesian Fact-Checked News Scraper - Main Execution CLI
Optimized for high-volume collection (e.g. 1,500+ articles) with:
- Automated pagination across portal archives
- Real-time progress tracking
- Periodic auto-exporting to JSON and CSV every 25 records
- Multi-threaded parallel workers
"""

import sys
import argparse
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from config import (
    LOG_FILE_PATH,
    JSON_OUTPUT_PATH,
    CSV_OUTPUT_PATH,
    DB_PATH,
    MAX_CONCURRENT_WORKERS
)
from scrapers import SCRAPER_REGISTRY
from processors.deduplicator import Deduplicator
from storage.database import Database
from storage.exporter import Exporter


def setup_logging(log_level: int = logging.INFO):
    """Sets up dual logging to console and scraper.log file."""
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)


def is_date_in_range(pub_date: Optional[str], from_date: Optional[str], to_date: Optional[str]) -> bool:
    """Checks whether the article publication date falls within the requested range."""
    if not pub_date:
        return True
    try:
        if from_date and pub_date < from_date:
            return False
        if to_date and pub_date > to_date:
            return False
        return True
    except Exception:
        return True


def run_scraper(
    source_key: str,
    mode: str,
    limit: int,
    from_date: Optional[str],
    to_date: Optional[str],
    workers: int,
    download_thumbnails: bool,
    db: Database,
    dedup: Deduplicator,
    exporter: Exporter
) -> List[Dict[str, Any]]:
    """Runs high-volume scraping pipeline across sources."""
    logger = logging.getLogger("ScraperRunner")
    
    # Determine which scrapers to instantiate
    if source_key in ("all", "factcheck"):
        # Focused sources: BBC, Tempo, Kompas, Liputan6, CNN only
        target_sources = ["bbc", "tempo", "kompas", "liputan6", "cnn"]
    else:
        if source_key not in SCRAPER_REGISTRY:
            logger.error(f"Unknown source '{source_key}'. Available: {', '.join(SCRAPER_REGISTRY.keys())}")
            return []
        target_sources = [source_key]

    scraped_articles: List[Dict[str, Any]] = []
    
    current_stored = db.count_articles()
    logger.info(f"Target total articles: {limit} (Currently in DB: {current_stored}) | Active sources: {len(target_sources)} | Parallel workers: {workers}")

    for s_name in target_sources:
        current_stored = db.count_articles()
        if current_stored >= limit:
            logger.info(f"Target of {limit} fact-checked articles already reached!")
            break

        needed = limit - current_stored
        discovery_limit = max(needed * 2, 600)

        scraper_cls = SCRAPER_REGISTRY[s_name]
        scraper = scraper_cls()
        logger.info(f"=== Starting Source: {scraper.source_name} (Seeking up to {needed} more articles) ===")

        # Step 1: Discover candidate URLs using multi-page pagination
        try:
            discovered_urls = scraper.discover_articles(mode=mode, limit=discovery_limit)
            logger.info(f"Discovered {len(discovered_urls)} candidate URLs from {scraper.source_name}")
        except Exception as e:
            logger.error(f"Error discovering URLs for {s_name}: {e}")
            continue

        if not discovered_urls:
            continue

        # Step 2: Parallel scraping of discovered URLs
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {
                executor.submit(scraper.scrape_article, url, download_thumbnails): url
                for url in discovered_urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article = future.result()
                    if not article:
                        continue

                    # STRICT REQUIREMENT: Only save articles with fact_check_status == 'TRUE'
                    if article.get("fact_check_status") != "TRUE":
                        continue

                    # Filter by date
                    if not is_date_in_range(article.get("publication_date"), from_date, to_date):
                        continue

                    # Deduplication check (URL, thumbnail, headline exact & similarity, caption similarity)
                    is_dup, dup_reason = dedup.is_duplicate(
                        article["article_url"],
                        article["headline"],
                        article.get("main_caption"),
                        article.get("thumbnail_url")
                    )
                    if is_dup:
                        logger.debug(f"Skipping duplicate: {dup_reason}")
                        continue

                    # Save to database
                    row_id = db.save_article(article)
                    if not row_id:
                        continue

                    # Register into deduplicator
                    dedup.register(
                        article["article_url"],
                        article["headline"],
                        article.get("main_caption"),
                        article.get("thumbnail_url")
                    )
                    article["id"] = row_id
                    scraped_articles.append(article)

                    count = db.count_articles()
                    progress_pct = min(100.0, (count / limit) * 100)
                    logger.info(
                        f"[{count}/{limit} - {progress_pct:.1f}%] "
                        f"[{article['source_name']}] [{article['verification_type']}] "
                        f"{article['headline'][:65]}..."
                    )

                    # Periodic incremental export every 25 articles
                    if count % 25 == 0:
                        all_current = db.get_articles(limit=limit)
                        exporter.export_all(all_current)
                        logger.info(f"==> Auto-exported {len(all_current)} articles to JSON and CSV.")

                    if count >= limit:
                        logger.info(f"Successfully reached target goal of {limit} articles!")
                        break

                except Exception as e:
                    logger.error(f"Error processing article {url}: {e}")

    return scraped_articles


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Indonesian Fact-Checked News Scraper (High-Volume Scale)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="Specify news source (kompas, tempo, detik, cnn, antara, turnbackhoax, liputan6, tirto, bbc, all). Default: all"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Target number of articles to scrape. Default: 1500"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["fact-check", "verified", "all"],
        help="Scraping mode: 'fact-check' (dedicated only), 'verified' (verified news), 'all' (both). Default: all"
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Filter articles published on or after YYYY-MM-DD"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Filter articles published on or before YYYY-MM-DD"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads. Default: 8"
    )
    parser.add_argument(
        "--download-thumbnails",
        action="store_true",
        help="Download and save thumbnail images to output/thumbnails/"
    )
    return parser.parse_args()


def print_summary(articles: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("               SCRAPING PROCESS COMPLETED")
    print("=" * 70)
    print(f"Total Articles Stored     : {len(articles)}")
    
    fact_checks = [a for a in articles if a.get("verification_type") == "DEDICATED_FACT_CHECK"]
    verified = [a for a in articles if a.get("verification_type") == "VERIFIED_NEWS"]
    print(f"Dedicated Fact-Checks     : {len(fact_checks)}")
    print(f"Verified News Articles    : {len(verified)}")

    # Breakdown by Source
    sources: Dict[str, int] = {}
    for a in articles:
        src = a.get("source_name", "Unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nBreakdown by Source:")
    for s, cnt in sources.items():
        print(f"  - {s:<25}: {cnt}")

    if fact_checks:
        print("\nFact-Check Verdict Breakdown:")
        verdicts: Dict[str, int] = {}
        for fc in fact_checks:
            st = fc.get("fact_check_status", "UNKNOWN")
            verdicts[st] = verdicts.get(st, 0) + 1
        for v, cnt in verdicts.items():
            print(f"  - {v:<15}: {cnt}")

    print("\nStorage Destinations:")
    print(f"  - SQLite Database : {DB_PATH}")
    print(f"  - JSON File       : {JSON_OUTPUT_PATH}")
    print(f"  - CSV File        : {CSV_OUTPUT_PATH}")
    print(f"  - Log File        : {LOG_FILE_PATH}")
    print("=" * 70 + "\n")


def main():
    args = parse_arguments()
    setup_logging()
    logger = logging.getLogger("Main")

    logger.info("Initializing Indonesian Fact-Checked News Scraper...")
    logger.info(f"Target: {args.limit} articles | Source: {args.source} | Mode: {args.mode} | Workers: {args.workers}")

    db = Database(DB_PATH)
    dedup = Deduplicator()
    exporter = Exporter(JSON_OUTPUT_PATH, CSV_OUTPUT_PATH)

    # Pre-populate deduplicator with existing URLs, thumbnails, headlines, and captions
    existing_records = db.get_articles(limit=5000)
    for rec in existing_records:
        dedup.register(
            rec["article_url"],
            rec["headline"],
            rec.get("main_caption"),
            rec.get("thumbnail_url")
        )
    logger.info(f"Loaded {len(existing_records)} existing records into deduplication memory.")

    # Run scraping
    scraped = run_scraper(
        source_key=args.source.lower(),
        mode=args.mode,
        limit=args.limit,
        from_date=args.from_date,
        to_date=args.to_date,
        workers=args.workers,
        download_thumbnails=args.download_thumbnails,
        db=db,
        dedup=dedup,
        exporter=exporter
    )

    # Final complete export
    all_articles = db.get_articles(
        source=args.source if args.source != "all" else None,
        mode=args.mode if args.mode != "all" else None,
        from_date=args.from_date,
        to_date=args.to_date,
        limit=args.limit
    )
    exporter.export_all(all_articles)
    print_summary(all_articles)


if __name__ == "__main__":
    main()
