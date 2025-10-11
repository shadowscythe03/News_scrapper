"""
News Aggregator ML Pipeline
A comprehensive system for news scraping, classification, and analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .scraping import NewsScraper, NewsArticle, DatabaseManager
from .classification import NewsClassifier
from .chatbot import ChatbotEngine, NewsAnalyzer
# Note: NewsScheduler moved to GitHub Actions automation

__all__ = [
    'NewsScraper',
    'NewsArticle', 
    'DatabaseManager',
    'NewsClassifier',
    'ChatbotEngine',
    'NewsAnalyzer'
]