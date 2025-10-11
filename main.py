#!/usr/bin/env python3
"""
Main entry point for the News Aggregator system
Provides a unified interface for running different components
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_scraping():
    """Run news scraping"""
    from src.scraping.news_scraper import NewsScraper
    
    print("🔍 Starting news scraping...")
    scraper = NewsScraper("configs/config.yaml")
    count = scraper.scrape_all_sources()
    print(f"✅ Scraped {count} new articles")

def run_classification():
    """Run article classification"""
    from src.classification.classifier import NewsClassifier
    
    print("🤖 Starting article classification...")
    classifier = NewsClassifier("configs/config.yaml")
    
    # Train models if they don't exist
    if not os.path.exists("models/sklearn_classifier.joblib"):
        print("📚 Training classification models...")
        classifier.train_models()
    
    # Load models and classify
    classifier.load_models()
    count = classifier.classify_articles_in_db()
    print(f"✅ Classified {count} articles")

def run_web_app():
    """Run the web application"""
    print("🌐 Starting web application...")
    os.chdir("src/web_app")
    subprocess.run([sys.executable, "app.py"])

# Scheduler functionality moved to GitHub Actions
# See .github/workflows/ml-pipeline.yml for automated scheduling

def run_dvc_pipeline():
    """Run DVC pipeline"""
    print("📊 Running DVC pipeline...")
    try:
        result = subprocess.run(['dvc', 'repro'], check=True)
        print("✅ DVC pipeline completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ DVC pipeline failed: {e}")
    except FileNotFoundError:
        print("❌ DVC not installed. Install it with: pip install dvc")

def setup_environment():
    """Setup the development environment"""
    print("🔧 Setting up development environment...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "logs",
        "evaluation"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Initialize DVC if not already done
    if not os.path.exists(".dvc"):
        try:
            subprocess.run(['dvc', 'init'], check=True)
            print("✅ Initialized DVC repository")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Could not initialize DVC (not installed or already initialized)")
    
    print("✅ Environment setup complete!")

def run_full_pipeline():
    """Run the complete pipeline"""
    print("🚀 Running full ML pipeline...")
    
    # Setup environment
    setup_environment()
    
    # Run scraping
    run_scraping()
    
    # Run classification
    run_classification()
    
    # Run DVC pipeline
    run_dvc_pipeline()
    
    print("✅ Full pipeline completed!")

def main():
    parser = argparse.ArgumentParser(description="News Aggregator System")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scraping command
    subparsers.add_parser('scrape', help='Run news scraping')
    
    # Classification command
    subparsers.add_parser('classify', help='Run article classification')
    
    # Web app command
    subparsers.add_parser('web', help='Run web application')
    
    # Note: Scheduler moved to GitHub Actions (see .github/workflows/ml-pipeline.yml)
    
    # DVC command
    subparsers.add_parser('dvc', help='Run DVC pipeline')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup development environment')
    
    # Full pipeline command
    subparsers.add_parser('pipeline', help='Run full ML pipeline')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'scrape':
            run_scraping()
        elif args.command == 'classify':
            run_classification()
        elif args.command == 'web':
            run_web_app()
        elif args.command == 'schedule':
            print("⚠️ Scheduler functionality moved to GitHub Actions!")
            print("📋 Check .github/workflows/ml-pipeline.yml for automated scheduling")
            print("🔧 Use 'python main.py pipeline' to run the full pipeline manually")
        elif args.command == 'dvc':
            run_dvc_pipeline()
        elif args.command == 'setup':
            setup_environment()
        elif args.command == 'pipeline':
            run_full_pipeline()
        else:
            print(f"Unknown command: {args.command}")
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())