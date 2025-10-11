#!/usr/bin/env python3
"""
Script to preprocess data for DVC pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import sqlite3
import pickle
import yaml
from sklearn.model_selection import train_test_split
import logging

def main():
    """Preprocess data for training"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)
        
        # Connect to database
        conn = sqlite3.connect("data/raw/articles.db")
        
        # Extract training data
        query = '''
            SELECT title, content, category 
            FROM articles 
            WHERE content IS NOT NULL 
            AND LENGTH(content) >= ?
            AND LENGTH(content) <= ?
        '''
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=[
                params['data_processing']['min_text_length'],
                params['data_processing']['max_text_length']
            ]
        )
        conn.close()
        
        logger.info(f"Extracted {len(df)} articles for training")
        
        if df.empty:
            logger.warning("No data found, creating synthetic data")
            df = create_synthetic_data(config['classification']['categories'])
        
        # Combine title and content
        df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
        
        # Basic text preprocessing
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
        df['text'] = df['text'].str.replace(r'\S*@\S*\s?', '', regex=True)
        
        # Filter valid categories
        valid_categories = config['classification']['categories']
        df = df[df['category'].isin(valid_categories)]
        
        # Split data
        train_df, test_df = train_test_split(
            df,
            test_size=params['data_processing']['test_size'],
            random_state=params['data_processing']['random_state'],
            stratify=df['category'] if len(df['category'].unique()) > 1 else None
        )
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        
        # Save training data
        train_df.to_csv("data/processed/training_data.csv", index=False)
        test_df.to_csv("data/processed/test_data.csv", index=False)
        
        # Save features metadata
        features_info = {
            'n_samples': len(df),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'categories': df['category'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean()
        }
        
        with open("data/processed/features.pkl", 'wb') as f:
            pickle.dump(features_info, f)
        
        logger.info(f"Saved training data: {len(train_df)} train, {len(test_df)} test samples")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        sys.exit(1)

def create_synthetic_data(categories):
    """Create synthetic training data"""
    synthetic_data = {
        'politics': [
            "Government announces new policy changes for economic recovery and growth",
            "Election results show surprising voter turnout across the country",
            "Political leaders meet to discuss international trade agreements",
            "Senate votes on controversial healthcare legislation reform",
        ],
        'technology': [
            "New artificial intelligence breakthrough revolutionizes data processing",
            "Tech giant releases innovative smartphone with advanced features",
            "Cybersecurity experts warn about emerging digital threats",
            "Software update brings enhanced performance and security improvements",
        ],
        'business': [
            "Stock market reaches record highs amid investor optimism",
            "Major corporation announces quarterly earnings exceeding expectations",
            "Economic indicators suggest steady growth in manufacturing sector",
            "Startup receives significant funding for expansion plans",
        ],
        'sports': [
            "Championship game draws millions of viewers worldwide",
            "Professional athlete breaks long-standing performance record",
            "Team announces new coaching staff for upcoming season",
            "Olympic training facility opens with state-of-the-art equipment",
        ],
        'health': [
            "Medical researchers discover potential treatment for rare disease",
            "Health officials recommend updated vaccination guidelines",
            "Study reveals benefits of regular exercise on mental wellness",
            "Hospital introduces innovative surgical technique for better outcomes",
        ],
        'science': [
            "Scientists make groundbreaking discovery about climate patterns",
            "Space mission successfully launches with advanced research equipment",
            "Research team develops new materials for renewable energy",
            "Archaeological findings shed light on ancient civilizations",
        ],
        'entertainment': [
            "Movie industry celebrates record-breaking box office weekend",
            "Popular streaming service announces exclusive content partnerships",
            "Music festival lineup features internationally acclaimed artists",
            "Television series wins multiple awards at industry ceremony",
        ]
    }
    
    data = []
    for category in categories:
        if category in synthetic_data:
            for text in synthetic_data[category]:
                data.append({
                    'title': text.split()[0:5],  # First 5 words as title
                    'content': text,
                    'category': category,
                    'text': text
                })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()