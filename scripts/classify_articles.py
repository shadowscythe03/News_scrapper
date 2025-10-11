#!/usr/bin/env python3
"""
Script to classify articles for DVC pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import sqlite3
from src.classification.classifier import NewsClassifier
import logging

def main():
    """Classify articles using trained models"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = NewsClassifier("configs/config.yaml")
        
        # Load trained models
        classifier.load_models()
        
        # Classify articles in database
        classified_count = classifier.classify_articles_in_db()
        
        # Export classified articles
        conn = sqlite3.connect("data/raw/articles.db")
        
        query = '''
            SELECT title, content, url, source, category, 
                   published_date, classified_category, sentiment_score
            FROM articles 
            WHERE classified_category IS NOT NULL
            ORDER BY published_date DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Save classified articles
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/classified_articles.csv", index=False)
        
        logger.info(f"Classified {classified_count} articles, exported {len(df)} articles")
        
    except Exception as e:
        logger.error(f"Error in classification pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()