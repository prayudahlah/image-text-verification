"""Reset all scraped data - JSON, CSV, SQLite DB, and logs."""
import sqlite3
import json

# 1. Reset JSON
with open("output/news.json", "w", encoding="utf-8") as f:
    json.dump({"articles": []}, f, indent=2)
print("Reset: output/news.json")

# 2. Reset CSV (headers only)
headers = "id,thumbnail_url,headline,main_caption,source_name,article_url,publication_date,verification_type,fact_check_status,fact_check_source,claim_checked\n"
with open("output/news.csv", "w", encoding="utf-8") as f:
    f.write(headers)
print("Reset: output/news.csv")

# 3. Reset SQLite database
conn = sqlite3.connect("storage/news.db")
cur = conn.cursor()
cur.execute("DELETE FROM articles")
try:
    cur.execute("DELETE FROM sqlite_sequence WHERE name='articles'")
except Exception:
    pass
conn.commit()
conn.close()
print("Reset: storage/news.db")

# 4. Clear log file
with open("logs/scraper.log", "w", encoding="utf-8") as f:
    f.write("")
print("Reset: logs/scraper.log")

print()
print("=== RESET COMPLETE - Semua data dikosongkan, siap scraping ulang ===")

