"""
News Scraper Module
Handles scraping from multiple news sources including RSS feeds and news APIs
"""

import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import time
import random
from typing import List, Dict, Optional
import sqlite3
import os
from dataclasses import dataclass
import hashlib
import yaml

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    content: str
    url: str
    source: str
    category: str
    published_date: datetime
    scraped_date: datetime
    article_hash: str
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'category': self.category,
            'published_date': self.published_date.isoformat(),
            'scraped_date': self.scraped_date.isoformat(),
            'article_hash': self.article_hash
        }

class DatabaseManager:
    """Manages SQLite database operations for news articles"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT UNIQUE,
                source TEXT,
                category TEXT,
                published_date TEXT,
                scraped_date TEXT,
                article_hash TEXT UNIQUE,
                classified_category TEXT,
                sentiment_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_published_date ON articles(published_date);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_category ON articles(category);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source ON articles(source);
        ''')
        
        conn.commit()
        conn.close()
    
    def save_article(self, article: NewsArticle) -> bool:
        """Save article to database, return True if saved, False if duplicate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (title, content, url, source, category, published_date, scraped_date, article_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.title,
                article.content,
                article.url,
                article.source,
                article.category,
                article.published_date.isoformat(),
                article.scraped_date.isoformat(),
                article.article_hash
            ))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success
            
        except Exception as e:
            logging.error(f"Error saving article: {e}")
            return False
    
    def get_articles(self, limit: int = 100, category: str = None, 
                    days_back: int = 7) -> List[Dict]:
        """Retrieve articles from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            base_query = '''
                SELECT title, content, url, source, category, published_date, 
                       classified_category, sentiment_score
                FROM articles 
                WHERE published_date >= date('now', '-{} days')
            '''.format(days_back)
            
            params = []
            if category:
                base_query += " AND (category = ? OR classified_category = ?)"
                params.extend([category, category])
            
            base_query += " ORDER BY published_date DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(base_query, params)
            columns = [desc[0] for desc in cursor.description]
            
            articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return articles
            
        except Exception as e:
            logging.error(f"Error retrieving articles: {e}")
            return []

class NewsScraper:
    """Main news scraping class"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_manager = DatabaseManager(self.config['database']['path'])
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['scraping']['user_agent']
        })
    
    def generate_hash(self, title: str, url: str) -> str:
        """Generate unique hash for article"""
        return hashlib.md5(f"{title}{url}".encode()).hexdigest()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        return text.strip()
    
    def scrape_rss_feed(self, feed_url: str, source_name: str, 
                       category: str) -> List[NewsArticle]:
        """Scrape articles from RSS feed"""
        articles = []
        
        try:
            self.logger.info(f"Scraping RSS feed: {source_name}")
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                self.logger.warning(f"No entries found in RSS feed: {feed_url}")
                return articles
            
            max_articles = self.config['scraping']['max_articles_per_source']
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Extract basic info
                    title = self.clean_text(entry.get('title', ''))
                    url = entry.get('link', '')
                    
                    if not title or not url:
                        continue
                    
                    # Parse published date
                    published_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])
                    
                    # Get full article content
                    content = self.get_article_content(url)
                    
                    # Create article object
                    article = NewsArticle(
                        title=title,
                        content=content,
                        url=url,
                        source=source_name,
                        category=category,
                        published_date=published_date,
                        scraped_date=datetime.now(),
                        article_hash=self.generate_hash(title, url)
                    )
                    
                    articles.append(article)
                    
                    # Rate limiting
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    self.logger.error(f"Error processing RSS entry: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error scraping RSS feed {source_name}: {e}")
        
        return articles
    
    def get_article_content(self, url: str) -> str:
        """Extract full article content from URL"""
        try:
            response = self.session.get(
                url, 
                timeout=self.config['scraping']['timeout']
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                element.decompose()
            
            # Try to find main content
            content_selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '.entry-content',
                'main',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body content
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            
            return self.clean_text(content)[:5000]  # Limit content length
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.debug(f"Article not found (404): {url}")
            else:
                self.logger.warning(f"HTTP error extracting content from {url}: {e}")
            return ""
        except Exception as e:
            self.logger.warning(f"Error extracting content from {url}: {e}")
            return ""
    
    def scrape_all_sources(self) -> int:
        """Scrape all configured news sources"""
        total_articles = 0
        
        self.logger.info("Starting news scraping process")
        
        # Scrape RSS feeds
        for source in self.config['news_sources']['rss_feeds']:
            articles = self.scrape_rss_feed(
                source['url'],
                source['name'],
                source['category']
            )
            
            # Save articles to database
            saved_count = 0
            for article in articles:
                if self.db_manager.save_article(article):
                    saved_count += 1
            
            total_articles += saved_count
            self.logger.info(f"Saved {saved_count} new articles from {source['name']}")
        
        self.logger.info(f"Scraping completed. Total new articles: {total_articles}")
        return total_articles
    
    def get_recent_articles(self, category: str = None, limit: int = 100) -> List[Dict]:
        """Get recent articles from database"""
        return self.db_manager.get_articles(category=category, limit=limit)

if __name__ == "__main__":
    # Test the scraper
    scraper = NewsScraper("configs/config.yaml")
    articles_count = scraper.scrape_all_sources()
    print(f"Scraped {articles_count} new articles")