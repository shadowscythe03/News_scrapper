# News Aggregator ML Pipeline

A comprehensive machine learning pipeline for news aggregation, classification, and analysis with an intelligent chatbot interface.

## 🚀 Features

- **Multi-source News Scraping**: Automated scraping from RSS feeds and news APIs
- **AI-Powered Classification**: CPU-optimized text classification using DistilBERT and traditional ML
- **Sentiment Analysis**: Real-time sentiment analysis of news articles
- **Intelligent Chatbot**: AI assistant for news summaries, mood analysis, and Q&A
- **Web Interface**: Modern Flask web application with responsive design
- **Data Versioning**: DVC integration for reproducible ML pipelines
- **Automated Scheduling**: Configurable automation for periodic updates
- **CPU-Optimized**: Designed to run efficiently without GPU requirements

## 📁 Project Structure

```
news_aggregator/
├── src/
│   ├── scraping/          # News scraping modules
│   ├── classification/    # ML classification system
│   ├── chatbot/          # AI chatbot engine
│   ├── web_app/          # Flask web application
│   └── automation/       # Scheduling and automation
├── data/
│   ├── raw/              # Raw scraped data
│   └── processed/        # Processed datasets
├── models/               # Trained ML models
├── configs/              # Configuration files
├── scripts/              # DVC pipeline scripts
├── logs/                 # Application logs
├── requirements.txt      # Python dependencies
├── dvc.yaml             # DVC pipeline definition
├── params.yaml          # Pipeline parameters
└── main.py              # Main entry point
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git (for version control)
- Optional: Virtual environment tool (venv, conda, etc.)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd news_aggregator
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment

```bash
python main.py setup
```

This will:
- Create necessary directories
- Initialize DVC repository
- Set up the basic project structure

### Step 5: Configure the System

Edit `configs/config.yaml` to customize:
- News sources (RSS feeds)
- Classification categories
- Web app settings
- Scraping intervals

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
python main.py pipeline
```

This will:
- Scrape news from configured sources
- Train classification models
- Classify all articles
- Set up DVC tracking

### 2. Start the Web Application

```bash
python main.py web
```

Visit `http://localhost:5000` to access the web interface.

### 3. Start Automated Scheduling

```bash
python main.py schedule
```

This will run continuous background tasks for periodic news updates.

## 📊 Usage

### Individual Commands

```bash
# Scrape news articles
python main.py scrape

# Train and run classification
python main.py classify

# Run DVC pipeline
python main.py dvc

# Start web app
python main.py web

# Start scheduler
python main.py schedule
```

### DVC Pipeline

The project includes a complete DVC pipeline for reproducible ML:

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro data_collection
dvc repro model_training
```

### Web Interface Features

1. **Home Page**: Latest news articles with sentiment indicators
2. **Category Views**: Filter articles by topic
3. **AI Chatbot**: Intelligent assistant for news analysis
4. **Statistics**: Real-time analytics and visualizations

### Chatbot Capabilities

The AI assistant can help with:

- **News Summaries**: "Summarize the latest news"
- **Sentiment Analysis**: "What's the current mood?"
- **Topic Analysis**: "What are the trending topics?"
- **Category-specific Queries**: "Show me technology news"
- **Headlines**: "What are the latest headlines?"

## ⚙️ Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
# Database settings
database:
  type: "sqlite"
  path: "data/news_database.db"

# News sources
news_sources:
  rss_feeds:
    - name: "BBC News"
      url: "http://feeds.bbci.co.uk/news/rss.xml"
      category: "general"

# Classification model
classification:
  model_name: "distilbert-base-uncased"
  categories:
    - "politics"
    - "technology"
    - "business"
    # ... more categories

# Web app settings
web_app:
  host: "0.0.0.0"
  port: 5000
  debug: true

# Scraping settings
scraping:
  interval_hours: 168  # 1 week
  max_articles_per_source: 50
```

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_key  # Optional for enhanced chatbot
```

## 🤖 Machine Learning Pipeline

### Classification Models

The system uses a hybrid approach:

1. **Traditional ML**: TF-IDF + Logistic Regression (fast, reliable)
2. **Transformer Model**: DistilBERT (higher accuracy, CPU-optimized)

### Model Training

Models are automatically trained when:
- No existing models are found
- Sufficient new training data is available
- Manual retraining is triggered

### Sentiment Analysis

Uses multiple approaches:
- **VADER Sentiment**: Rule-based sentiment analysis
- **TextBlob**: Statistical sentiment analysis
- Combined scoring for robust results

## 📈 Data Pipeline

### Data Flow

1. **Raw Data**: News articles scraped from sources
2. **Processing**: Text cleaning, deduplication, validation
3. **Feature Extraction**: TF-IDF vectors, embeddings
4. **Model Training**: Classification and sentiment models
5. **Inference**: Real-time classification of new articles
6. **Storage**: Structured database with metadata

### DVC Stages

1. **data_collection**: Scrape news from sources
2. **data_preprocessing**: Clean and prepare training data
3. **model_training**: Train classification models
4. **classification**: Apply models to new articles
5. **evaluation**: Generate metrics and reports

## 🌐 Web Application

### Architecture

- **Backend**: Flask with SQLite database
- **Frontend**: Bootstrap 5 with responsive design
- **API**: RESTful endpoints for all functionalities
- **Real-time Updates**: AJAX for dynamic content

### API Endpoints

```
GET  /                    # Main page
GET  /category/<name>     # Category view
GET  /chatbot            # Chatbot interface
GET  /api/articles       # Articles API
GET  /api/stats          # Statistics API
POST /api/scrape         # Trigger scraping
POST /api/chat           # Chatbot API
```

## 🤖 Chatbot Engine

### Natural Language Processing

- **Intent Detection**: Pattern-based intent recognition
- **Entity Extraction**: Category and topic extraction
- **Context Understanding**: Multi-turn conversation support

### Response Generation

- **Template-based**: Structured responses for common queries
- **Data-driven**: Dynamic responses based on current news data
- **Personalized**: Contextual responses based on query specifics

## 📅 Automation & Scheduling

### Scheduled Tasks

- **News Scraping**: Configurable intervals (default: weekly)
- **Model Retraining**: Weekly with sufficient new data
- **DVC Pipeline**: Daily data versioning
- **Health Checks**: System monitoring every 6 hours

### Running as Service

#### Windows Service

Create a batch file to run as service:

```batch
@echo off
cd /d "C:\path\to\news_aggregator"
python main.py schedule --daemon
```

#### Linux Service (systemd)

Create `/etc/systemd/system/news-aggregator.service`:

```ini
[Unit]
Description=News Aggregator Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/news_aggregator
ExecStart=/path/to/python main.py schedule --daemon
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔧 Development

### Adding New News Sources

Edit `configs/config.yaml`:

```yaml
news_sources:
  rss_feeds:
    - name: "New Source"
      url: "https://example.com/rss"
      category: "technology"
```

### Adding New Categories

1. Update `configs/config.yaml`
2. Retrain classification models
3. Update chatbot responses if needed

### Extending the Chatbot

Add new intent patterns in `src/chatbot/chatbot_engine.py`:

```python
self.intent_patterns = {
    'new_intent': [
        r'pattern1', r'pattern2'
    ]
}
```

## 📊 Monitoring & Logging

### Log Files

- `logs/scheduler.log`: Automation and scheduling
- Web app logs: Console output
- DVC logs: Pipeline execution

### Health Monitoring

The system includes built-in health checks:

```bash
python -c "from src.automation.scheduler import NewsScheduler; s = NewsScheduler('configs/config.yaml'); print(s.health_check())"
```

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**: Ensure models are trained first
2. **Database locked**: Close all connections and restart
3. **Memory issues**: Reduce batch sizes in config
4. **Scraping failures**: Check RSS feed URLs and network connectivity

### Debug Mode

Enable debug logging in `configs/config.yaml`:

```yaml
logging:
  level: "DEBUG"
```

### Performance Optimization

For CPU-constrained environments:
- Reduce `batch_size` in training config
- Limit `max_articles_per_source`
- Use only sklearn models (disable transformer)

## 📄 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

## 🔄 Version History

- **v1.0.0**: Initial release with complete ML pipeline
- Features: News scraping, classification, chatbot, web interface
- CPU-optimized for broad compatibility

---

**Happy News Aggregating! 📰🤖**