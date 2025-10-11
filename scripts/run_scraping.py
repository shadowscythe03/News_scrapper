#!/usr/bin/env python3
"""
Script to run news scraping for DVC pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.scraping.news_scraper import NewsScraper
import logging

def main():
    """Run news scraping"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize scraper
        scraper = NewsScraper("configs/config.yaml")
        
        # Scrape articles
        articles_count = scraper.scrape_all_sources()
        
        logger.info(f"Successfully scraped {articles_count} new articles")
        
        # Copy database to raw data directory for DVC tracking
        import shutil
        src_db = scraper.config['database']['path']
        dst_db = "data/raw/articles.db"
        
        os.makedirs(os.path.dirname(dst_db), exist_ok=True)
        if os.path.exists(src_db):
            shutil.copy2(src_db, dst_db)
            logger.info(f"Copied database to {dst_db}")
        
        return articles_count
        
    except Exception as e:
        logger.error(f"Error in scraping pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()