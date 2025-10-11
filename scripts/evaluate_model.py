#!/usr/bin/env python3
"""
Evaluation script for model performance
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import sqlite3
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.classification.classifier import NewsClassifier

def evaluate_model():
    """Evaluate classification model performance"""
    
    # Load classifier
    classifier = NewsClassifier("configs/config.yaml")
    classifier.load_models()
    
    # Get test data from database
    conn = sqlite3.connect("data/news_database.db")
    
    query = '''
        SELECT title, content, category, classified_category
        FROM articles 
        WHERE classified_category IS NOT NULL
        AND category IS NOT NULL
        AND content IS NOT NULL
        LIMIT 1000
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No data available for evaluation")
        return
    
    # Prepare data
    y_true = df['category'].values
    y_pred = df['classified_category'].values
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Save metrics
    os.makedirs("evaluation", exist_ok=True)
    
    with open("evaluation/metrics.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg'],
            'classification_report': report
        }, f, indent=2)
    
    # Generate confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    categories = sorted(list(set(y_true) | set(y_pred)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("evaluation/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save text report
    with open("evaluation/classification_report.txt", 'w') as f:
        f.write(f"Model Evaluation Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Dataset size: {len(df)} articles\n\n")
        f.write(classification_report(y_true, y_pred))
    
    print(f"✅ Evaluation completed. Results saved to evaluation/")
    print(f"📊 Accuracy: {report['accuracy']:.3f}")
    print(f"📈 Macro F1: {report['macro avg']['f1-score']:.3f}")

if __name__ == "__main__":
    evaluate_model()