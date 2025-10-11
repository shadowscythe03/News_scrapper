"""
Initialization file for scraping module
"""

from .news_scraper import NewsScraper, NewsArticle, DatabaseManager

__all__ = ['NewsScraper', 'NewsArticle', 'DatabaseManager']