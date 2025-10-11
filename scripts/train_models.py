#!/usr/bin/env python3
"""
Script to train models for DVC pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
from src.classification.classifier import NewsClassifier
import logging

def main():
    """Train classification models"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize classifier
        classifier = NewsClassifier("configs/config.yaml")
        
        # Load training data
        train_df = pd.read_csv("data/processed/training_data.csv")
        
        logger.info(f"Training with {len(train_df)} samples")
        
        # Train models
        classifier.train_sklearn_model(train_df)
        
        # Train transformer model if enough data
        if len(train_df) > 50:
            try:
                classifier.train_transformer_model(train_df)
                transformer_trained = True
            except Exception as e:
                logger.warning(f"Transformer training failed: {e}")
                transformer_trained = False
        else:
            transformer_trained = False
        
        # Create metrics
        metrics = {
            "training_samples": len(train_df),
            "categories": len(train_df['category'].unique()),
            "transformer_trained": transformer_trained,
            "sklearn_trained": True
        }
        
        # Save metrics
        os.makedirs("models", exist_ok=True)
        with open("models/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()