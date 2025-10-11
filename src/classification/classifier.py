"""
News Classification Module
Implements CPU-optimized text classification for news articles
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import os
import logging
import yaml
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime
import joblib
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsClassifier:
    """News article classifier using both traditional ML and transformer models"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.categories = self.config['classification']['categories']
        self.label_encoder = LabelEncoder()
        self.tfidf_model = None
        self.sklearn_model = None
        self.transformer_model = None
        self.tokenizer = None
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Database connection
        self.db_path = self.config['database']['path']
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for classification"""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Remove URLs, emails, and mentions (basic cleaning)
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        return text.strip()
    
    def create_training_data(self) -> pd.DataFrame:
        """Create training dataset from scraped articles"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get articles with their original categories as ground truth
            query = '''
                SELECT title, content, category 
                FROM articles 
                WHERE content IS NOT NULL 
                AND LENGTH(content) > 100
                AND category IN ({})
            '''.format(','.join(['?' for _ in self.categories]))
            
            df = pd.read_sql_query(query, conn, params=self.categories)
            conn.close()
            
            if df.empty:
                self.logger.warning("No training data found in database")
                return self._create_synthetic_training_data()
            
            # Combine title and content for classification
            df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
            df['text'] = df['text'].apply(self.preprocess_text)
            
            # Filter out very short texts
            df = df[df['text'].str.len() > 50]
            
            self.logger.info(f"Created training dataset with {len(df)} samples")
            return df[['text', 'category']]
            
        except Exception as e:
            self.logger.error(f"Error creating training data: {e}")
            return self._create_synthetic_training_data()
    
    def _create_synthetic_training_data(self) -> pd.DataFrame:
        """Create synthetic training data if no real data is available"""
        self.logger.info("Creating synthetic training data")
        
        synthetic_data = {
            'politics': [
                "Government announces new policy changes for economic recovery",
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
        for category, texts in synthetic_data.items():
            for text in texts:
                data.append({'text': text, 'category': category})
        
        return pd.DataFrame(data)
    
    def train_sklearn_model(self, df: pd.DataFrame) -> None:
        """Train traditional ML model using TF-IDF and Logistic Regression"""
        self.logger.info("Training sklearn classification model")
        
        # Prepare data
        X = df['text']
        y = df['category']
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # TF-IDF vectorization
        self.tfidf_model = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = self.tfidf_model.fit_transform(X_train)
        X_test_tfidf = self.tfidf_model.transform(X_test)
        
        # Train model
        self.sklearn_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        
        self.sklearn_model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.sklearn_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Sklearn model accuracy: {accuracy:.3f}")
        
        # Save models
        joblib.dump(self.sklearn_model, os.path.join(self.model_dir, 'sklearn_classifier.joblib'))
        joblib.dump(self.tfidf_model, os.path.join(self.model_dir, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, 'label_encoder.joblib'))
    
    def train_transformer_model(self, df: pd.DataFrame) -> None:
        """Train transformer model (CPU-optimized)"""
        self.logger.info("Training transformer classification model")
        
        try:
            from datasets import Dataset
            
            # Prepare data
            X = df['text'].tolist()
            y = df['category'].tolist()
            
            # Encode labels
            label_to_id = {label: idx for idx, label in enumerate(self.categories)}
            id_to_label = {idx: label for label, idx in label_to_id.items()}
            
            y_encoded = [label_to_id.get(label, 0) for label in y]
            
            # Create dataset
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Initialize tokenizer and model
            model_name = self.config['classification']['model_name']
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.categories),
                id2label=id_to_label,
                label2id=label_to_id
            )
            
            # Tokenize data
            def tokenize_function(texts):
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            
            train_encodings = tokenize_function(train_texts)
            val_encodings = tokenize_function(val_texts)
            
            train_dataset = Dataset.from_dict({
                **train_encodings,
                'labels': train_labels
            })
            
            val_dataset = Dataset.from_dict({
                **val_encodings,
                'labels': val_labels
            })
            
            # Training arguments (CPU optimized)
            training_args = TrainingArguments(
                output_dir=os.path.join(self.model_dir, 'transformer_model'),
                num_train_epochs=self.config['classification']['training']['epochs'],
                per_device_train_batch_size=self.config['classification']['training']['batch_size'],
                per_device_eval_batch_size=self.config['classification']['training']['batch_size'],
                learning_rate=self.config['classification']['training']['learning_rate'],
                warmup_steps=100,
                logging_dir=os.path.join(self.model_dir, 'logs'),
                logging_steps=10,
                eval_strategy="epoch",  # Changed from evaluation_strategy
                save_strategy="epoch",
                load_best_model_at_end=True,
                use_cpu=True,  # Force CPU usage
                dataloader_num_workers=0,  # Avoid multiprocessing issues on some systems
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(os.path.join(self.model_dir, 'transformer_model'))
            
            self.logger.info("Transformer model training completed")
            
        except ImportError:
            self.logger.warning("datasets library not available, skipping transformer training")
        except Exception as e:
            self.logger.error(f"Error training transformer model: {e}")
    
    def load_models(self) -> None:
        """Load trained models"""
        try:
            # Load sklearn models
            sklearn_path = os.path.join(self.model_dir, 'sklearn_classifier.joblib')
            tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
            encoder_path = os.path.join(self.model_dir, 'label_encoder.joblib')
            
            if all(os.path.exists(p) for p in [sklearn_path, tfidf_path, encoder_path]):
                self.sklearn_model = joblib.load(sklearn_path)
                self.tfidf_model = joblib.load(tfidf_path)
                self.label_encoder = joblib.load(encoder_path)
                self.logger.info("Loaded sklearn models")
            
            # Load transformer model
            transformer_path = os.path.join(self.model_dir, 'transformer_model')
            if os.path.exists(transformer_path):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
                    self.transformer_model = pipeline(
                        "text-classification",
                        model=transformer_path,
                        tokenizer=transformer_path,
                        device=-1  # CPU
                    )
                    self.logger.info("Loaded transformer model")
                except Exception as e:
                    self.logger.warning(f"Could not load transformer model: {e}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def classify_text(self, text: str) -> Dict:
        """Classify a single text using available models"""
        text = self.preprocess_text(text)
        result = {
            'text': text,
            'sklearn_prediction': None,
            'transformer_prediction': None,
            'final_prediction': None,
            'confidence': 0.0,
            'sentiment': self.analyze_sentiment(text)
        }
        
        # Sklearn prediction
        if self.sklearn_model and self.tfidf_model:
            try:
                text_tfidf = self.tfidf_model.transform([text])
                sklearn_pred = self.sklearn_model.predict(text_tfidf)[0]
                sklearn_proba = self.sklearn_model.predict_proba(text_tfidf)[0]
                
                result['sklearn_prediction'] = self.label_encoder.inverse_transform([sklearn_pred])[0]
                result['sklearn_confidence'] = float(np.max(sklearn_proba))
                
            except Exception as e:
                self.logger.error(f"Error in sklearn prediction: {e}")
        
        # Transformer prediction
        if self.transformer_model:
            try:
                transformer_result = self.transformer_model(text)
                result['transformer_prediction'] = transformer_result[0]['label']
                result['transformer_confidence'] = transformer_result[0]['score']
                
            except Exception as e:
                self.logger.error(f"Error in transformer prediction: {e}")
        
        # Final prediction (prefer transformer if available)
        if result['transformer_prediction']:
            result['final_prediction'] = result['transformer_prediction']
            result['confidence'] = result['transformer_confidence']
        elif result['sklearn_prediction']:
            result['final_prediction'] = result['sklearn_prediction']
            result['confidence'] = result['sklearn_confidence']
        else:
            result['final_prediction'] = 'general'
            result['confidence'] = 0.5
        
        return result
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        try:
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            
            return {
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.33,
                'vader_negative': 0.33,
                'vader_neutral': 0.33,
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.5
            }
    
    def classify_articles_in_db(self) -> int:
        """Classify all unclassified articles in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get unclassified articles
            cursor.execute('''
                SELECT id, title, content 
                FROM articles 
                WHERE classified_category IS NULL 
                AND content IS NOT NULL
            ''')
            
            articles = cursor.fetchall()
            classified_count = 0
            
            for article_id, title, content in articles:
                text = f"{title} {content}"
                result = self.classify_text(text)
                
                # Update database
                cursor.execute('''
                    UPDATE articles 
                    SET classified_category = ?, 
                        sentiment_score = ?
                    WHERE id = ?
                ''', (
                    result['final_prediction'],
                    result['sentiment']['vader_compound'],
                    article_id
                ))
                
                classified_count += 1
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Classified {classified_count} articles")
            return classified_count
            
        except Exception as e:
            self.logger.error(f"Error classifying articles: {e}")
            return 0
    
    def train_models(self) -> None:
        """Train both sklearn and transformer models"""
        df = self.create_training_data()
        
        if df.empty:
            self.logger.error("No training data available")
            return
        
        self.train_sklearn_model(df)
        
        # Only train transformer if we have enough data
        if len(df) > 50:
            self.train_transformer_model(df)
        else:
            self.logger.info("Insufficient data for transformer training")

if __name__ == "__main__":
    # Test the classifier
    classifier = NewsClassifier("configs/config.yaml")
    
    # Train models if they don't exist
    if not os.path.exists("models/sklearn_classifier.joblib"):
        classifier.train_models()
    
    # Load models
    classifier.load_models()
    
    # Test classification
    test_text = "The government announced new economic policies to boost market recovery"
    result = classifier.classify_text(test_text)
    print(f"Classification result: {result}")
    
    # Classify articles in database
    classified_count = classifier.classify_articles_in_db()
    print(f"Classified {classified_count} articles")