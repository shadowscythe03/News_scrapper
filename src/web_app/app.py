"""
Flask Web Application for News Aggregator
Displays classified news articles with live updates and chatbot interface
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import os
import sys
import yaml
import logging
from typing import List, Dict, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.scraping.news_scraper import NewsScraper
from src.classification.classifier import NewsClassifier
from src.chatbot.chatbot_engine import ChatbotEngine

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables for production
    config['web_app']['host'] = os.environ.get('WEB_APP_HOST', config['web_app']['host'])
    config['web_app']['port'] = int(os.environ.get('WEB_APP_PORT', config['web_app']['port']))
    config['web_app']['debug'] = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    return app, config

app, config = create_app()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format=config['logging']['format']
)
logger = logging.getLogger(__name__)

# Initialize components
db_path = os.path.join(os.path.dirname(__file__), '..', '..', config['database']['path'])
scraper = None
classifier = None
chatbot = None

def init_components():
    """Initialize scraper, classifier, and chatbot"""
    global scraper, classifier, chatbot
    try:
        scraper = NewsScraper(config_path)
        classifier = NewsClassifier(config_path)
        classifier.load_models()
        chatbot = ChatbotEngine(config_path)
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Main page showing latest news"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get latest articles
        cursor.execute('''
            SELECT title, content, url, source, category, classified_category,
                   published_date, sentiment_score
            FROM articles 
            WHERE published_date >= date('now', '-7 days')
            ORDER BY published_date DESC 
            LIMIT ?
        ''', (config['web_app']['articles_per_page'],))
        
        articles = [dict(row) for row in cursor.fetchall()]
        
        # Get category counts
        cursor.execute('''
            SELECT COALESCE(classified_category, category) as cat, COUNT(*) as count
            FROM articles 
            WHERE published_date >= date('now', '-7 days')
            GROUP BY COALESCE(classified_category, category)
            ORDER BY count DESC
        ''')
        
        categories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return render_template('index.html', 
                             articles=articles, 
                             categories=categories,
                             total_articles=len(articles))
        
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('error.html', error=str(e))

@app.route('/category/<category_name>')
def category_view(category_name):
    """View articles by category"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, content, url, source, category, classified_category,
                   published_date, sentiment_score
            FROM articles 
            WHERE (category = ? OR classified_category = ?)
            AND published_date >= date('now', '-7 days')
            ORDER BY published_date DESC 
            LIMIT ?
        ''', (category_name, category_name, config['web_app']['articles_per_page']))
        
        articles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return render_template('category.html', 
                             articles=articles, 
                             category=category_name,
                             total_articles=len(articles))
        
    except Exception as e:
        logger.error(f"Error in category route: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/articles')
def api_articles():
    """API endpoint for articles"""
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 20))
        days_back = int(request.args.get('days', 7))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        base_query = '''
            SELECT title, content, url, source, category, classified_category,
                   published_date, sentiment_score
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
        articles = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'articles': articles,
            'count': len(articles),
            'category': category
        })
        
    except Exception as e:
        logger.error(f"Error in API articles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total articles
        cursor.execute('SELECT COUNT(*) as total FROM articles')
        total = cursor.fetchone()['total']
        
        # Articles by category
        cursor.execute('''
            SELECT COALESCE(classified_category, category) as cat, COUNT(*) as count
            FROM articles 
            WHERE published_date >= date('now', '-7 days')
            GROUP BY COALESCE(classified_category, category)
            ORDER BY count DESC
        ''')
        categories = [dict(row) for row in cursor.fetchall()]
        
        # Articles by source
        cursor.execute('''
            SELECT source, COUNT(*) as count
            FROM articles 
            WHERE published_date >= date('now', '-7 days')
            GROUP BY source
            ORDER BY count DESC
        ''')
        sources = [dict(row) for row in cursor.fetchall()]
        
        # Sentiment distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN sentiment_score > 0.1 THEN 'positive'
                    WHEN sentiment_score < -0.1 THEN 'negative'
                    ELSE 'neutral'
                END as sentiment,
                COUNT(*) as count
            FROM articles 
            WHERE sentiment_score IS NOT NULL
            AND published_date >= date('now', '-7 days')
            GROUP BY sentiment
        ''')
        sentiment = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total_articles': total,
            'categories': categories,
            'sources': sources,
            'sentiment': sentiment
        })
        
    except Exception as e:
        logger.error(f"Error in API stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/scrape', methods=['POST'])
def api_scrape():
    """API endpoint to trigger scraping"""
    try:
        if not scraper:
            return jsonify({'error': 'Scraper not initialized'}), 500
        
        articles_count = scraper.scrape_all_sources()
        
        # Classify new articles if classifier is available
        if classifier:
            classified_count = classifier.classify_articles_in_db()
            return jsonify({
                'message': f'Scraped {articles_count} new articles, classified {classified_count} articles',
                'scraped': articles_count,
                'classified': classified_count
            })
        else:
            return jsonify({
                'message': f'Scraped {articles_count} new articles',
                'scraped': articles_count
            })
        
    except Exception as e:
        logger.error(f"Error in API scrape: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot')
def chatbot_page():
    """Chatbot interface page"""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chatbot"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Simple chatbot responses (to be enhanced with actual NLP)
        response = generate_chatbot_response(message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat API: {e}")
        return jsonify({'error': str(e)}), 500

def generate_chatbot_response(message: str) -> str:
    """Generate chatbot response using advanced chatbot engine"""
    try:
        if chatbot:
            return chatbot.generate_response(message)
        else:
            return "The chatbot is not available right now. Please try again later."
    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return "I'm having trouble processing your request right now. Please try again later."

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    init_components()
    app.run(
        host=config['web_app']['host'],
        port=config['web_app']['port'],
        debug=config['web_app']['debug']
    )

# For production deployment (Gunicorn, etc.)
def create_production_app():
    """Create app for production deployment"""
    production_app, _ = create_app()
    with production_app.app_context():
        init_components()
    return production_app