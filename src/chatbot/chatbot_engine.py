"""
Advanced Chatbot Module for News Aggregator
Provides intelligent conversation and analysis capabilities
"""

import os
import sys
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import re
import yaml
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from collections import Counter
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class NewsAnalyzer:
    """Analyzes news data for chatbot responses"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def get_articles(self, days_back: int = 7, category: str = None, 
                    limit: int = 100) -> List[Dict]:
        """Get articles from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            base_query = '''
                SELECT title, content, url, source, category, classified_category,
                       published_date, sentiment_score
                FROM articles 
                WHERE published_date >= date('now', '-{} days')
                AND content IS NOT NULL
                AND LENGTH(content) > 50
            '''.format(days_back)
            
            params = []
            if category:
                base_query += " AND (category = ? OR classified_category = ?)"
                params.extend([category, category])
            
            base_query += " ORDER BY published_date DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(base_query, params)
            columns = [desc[0] for desc in cursor.description]
            articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return articles
            
        except Exception as e:
            logging.error(f"Error getting articles: {e}")
            return []
    
    def analyze_sentiment_trends(self, days_back: int = 7) -> Dict:
        """Analyze sentiment trends over time"""
        articles = self.get_articles(days_back=days_back)
        
        if not articles:
            return {"error": "No articles found for analysis"}
        
        sentiment_scores = [a['sentiment_score'] for a in articles if a['sentiment_score'] is not None]
        
        if not sentiment_scores:
            return {"error": "No sentiment data available"}
        
        positive = len([s for s in sentiment_scores if s > 0.1])
        negative = len([s for s in sentiment_scores if s < -0.1])
        neutral = len(sentiment_scores) - positive - negative
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Determine overall mood
        if avg_sentiment > 0.2:
            mood = "very positive"
        elif avg_sentiment > 0.05:
            mood = "slightly positive"
        elif avg_sentiment < -0.2:
            mood = "very negative"
        elif avg_sentiment < -0.05:
            mood = "slightly negative"
        else:
            mood = "neutral"
        
        return {
            "total_articles": len(articles),
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "positive_percent": round((positive / len(sentiment_scores)) * 100, 1),
            "negative_percent": round((negative / len(sentiment_scores)) * 100, 1),
            "neutral_percent": round((neutral / len(sentiment_scores)) * 100, 1),
            "average_sentiment": round(avg_sentiment, 3),
            "overall_mood": mood
        }
    
    def get_top_topics(self, days_back: int = 7, limit: int = 10) -> List[Dict]:
        """Get top topics/categories"""
        articles = self.get_articles(days_back=days_back)
        
        categories = []
        for article in articles:
            cat = article['classified_category'] or article['category'] or 'general'
            categories.append(cat)
        
        category_counts = Counter(categories)
        
        return [
            {"topic": topic, "count": count}
            for topic, count in category_counts.most_common(limit)
        ]
    
    def get_news_summary(self, days_back: int = 7, limit: int = 10) -> Dict:
        """Generate comprehensive news summary"""
        articles = self.get_articles(days_back=days_back, limit=limit)
        
        if not articles:
            return {"error": "No articles found for summary"}
        
        # Analyze topics
        topics = self.get_top_topics(days_back=days_back)
        
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment_trends(days_back=days_back)
        
        # Get top sources
        sources = Counter([a['source'] for a in articles])
        top_sources = sources.most_common(5)
        
        # Extract key headlines
        headlines = [a['title'] for a in articles[:5]]
        
        return {
            "period": f"last {days_back} days",
            "total_articles": len(articles),
            "top_topics": topics[:5],
            "sentiment_analysis": sentiment_analysis,
            "top_sources": [{"source": s[0], "count": s[1]} for s in top_sources],
            "key_headlines": headlines
        }

class ChatbotEngine:
    """Main chatbot engine with natural language processing"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = os.path.join(
            os.path.dirname(config_path), 
            self.config['database']['path']
        )
        
        self.analyzer = NewsAnalyzer(self.db_path)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns
        self.intent_patterns = {
            'summary': [
                r'summar[yi]', r'overview', r'recap', r'what.*happen',
                r'latest.*news', r'recent.*news'
            ],
            'sentiment': [
                r'mood', r'sentiment', r'feel', r'emotion', r'atmosphere',
                r'positive', r'negative', r'happy', r'sad'
            ],
            'topics': [
                r'topic', r'category', r'subject', r'about.*what',
                r'main.*story', r'trending'
            ],
            'specific_category': [
                r'politics?', r'technology?', r'business', r'sports?',
                r'health', r'science', r'entertainment'
            ],
            'headlines': [
                r'headline', r'title', r'latest', r'breaking',
                r'top.*story', r'main.*news'
            ],
            'help': [
                r'help', r'what.*do', r'how.*work', r'commands?',
                r'can.*you', r'able.*to'
            ]
        }
    
    def detect_intent(self, message: str) -> Tuple[str, float]:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return 'general', 0.0
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent] / len(message.split())
        
        return best_intent, confidence
    
    def extract_category(self, message: str) -> Optional[str]:
        """Extract specific category from message"""
        message_lower = message.lower()
        
        category_keywords = {
            'politics': ['politic', 'government', 'election', 'vote'],
            'technology': ['tech', 'computer', 'software', 'ai', 'digital'],
            'business': ['business', 'market', 'stock', 'economy', 'finance'],
            'sports': ['sport', 'game', 'team', 'player', 'match'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'disease'],
            'science': ['science', 'research', 'study', 'discovery'],
            'entertainment': ['entertainment', 'movie', 'music', 'celebrity']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
        
        return None
    
    def generate_response(self, message: str) -> str:
        """Generate chatbot response based on message"""
        try:
            intent, confidence = self.detect_intent(message)
            self.logger.info(f"Detected intent: {intent} (confidence: {confidence:.2f})")
            
            if intent == 'summary':
                return self._generate_summary_response(message)
            elif intent == 'sentiment':
                return self._generate_sentiment_response(message)
            elif intent == 'topics':
                return self._generate_topics_response(message)
            elif intent == 'headlines':
                return self._generate_headlines_response(message)
            elif intent == 'specific_category':
                category = self.extract_category(message)
                if category:
                    return self._generate_category_response(category)
                else:
                    return self._generate_topics_response(message)
            elif intent == 'help':
                return self._generate_help_response()
            else:
                return self._generate_general_response(message)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I'm having trouble processing your request right now. Please try again later."
    
    def _generate_summary_response(self, message: str) -> str:
        """Generate news summary response"""
        try:
            summary = self.analyzer.get_news_summary()
            
            if "error" in summary:
                return "I don't have enough recent news data to provide a summary. Try refreshing the news first."
            
            response = f"""📊 **News Summary for the {summary['period']}**

🗞️ **Total Articles:** {summary['total_articles']}

📈 **Top Topics:**"""
            
            for topic in summary['top_topics']:
                response += f"\n• {topic['topic'].title()}: {topic['count']} articles"
            
            sentiment = summary['sentiment_analysis']
            response += f"""

😊 **Sentiment Analysis:**
• Positive: {sentiment['positive_percent']}% ({sentiment['positive_count']} articles)
• Negative: {sentiment['negative_percent']}% ({sentiment['negative_count']} articles)  
• Neutral: {sentiment['neutral_percent']}% ({sentiment['neutral_count']} articles)
• Overall mood: {sentiment['overall_mood']}

📰 **Top Sources:**"""
            
            for source in summary['top_sources']:
                response += f"\n• {source['source']}: {source['count']} articles"
            
            response += "\n\n🔥 **Key Headlines:**"
            for i, headline in enumerate(summary['key_headlines'], 1):
                response += f"\n{i}. {headline}"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in summary response: {e}")
            return "I encountered an error while generating the news summary."
    
    def _generate_sentiment_response(self, message: str) -> str:
        """Generate sentiment analysis response"""
        try:
            sentiment = self.analyzer.analyze_sentiment_trends()
            
            if "error" in sentiment:
                return "I don't have enough sentiment data to analyze the current mood."
            
            mood_emoji = {
                "very positive": "😄",
                "slightly positive": "🙂", 
                "neutral": "😐",
                "slightly negative": "😕",
                "very negative": "😟"
            }
            
            emoji = mood_emoji.get(sentiment['overall_mood'], "😐")
            
            response = f"""{emoji} **Current News Sentiment Analysis**

📊 **Sentiment Distribution:**
• 😊 Positive: {sentiment['positive_percent']}% ({sentiment['positive_count']} articles)
• 😔 Negative: {sentiment['negative_percent']}% ({sentiment['negative_count']} articles)
• 😐 Neutral: {sentiment['neutral_percent']}% ({sentiment['neutral_count']} articles)

📈 **Overall Assessment:**
The current news climate appears to be **{sentiment['overall_mood']}** with an average sentiment score of {sentiment['average_sentiment']}.

Based on {sentiment['total_articles']} recent articles, the general mood in the news suggests a **{sentiment['overall_mood']}** outlook on current events."""
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in sentiment response: {e}")
            return "I encountered an error while analyzing the news sentiment."
    
    def _generate_topics_response(self, message: str) -> str:
        """Generate topics/categories response"""
        try:
            topics = self.analyzer.get_top_topics()
            
            if not topics:
                return "I don't have enough data to show trending topics right now."
            
            response = "🏷️ **Trending News Topics:**\n\n"
            
            for i, topic in enumerate(topics, 1):
                percentage = (topic['count'] / sum(t['count'] for t in topics)) * 100
                response += f"{i}. **{topic['topic'].title()}**: {topic['count']} articles ({percentage:.1f}%)\n"
            
            response += f"\nThese are the most discussed topics based on recent news coverage."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in topics response: {e}")
            return "I encountered an error while analyzing news topics."
    
    def _generate_headlines_response(self, message: str) -> str:
        """Generate headlines response"""
        try:
            articles = self.analyzer.get_articles(limit=10)
            
            if not articles:
                return "I don't have any recent headlines to show you."
            
            response = "📰 **Latest Headlines:**\n\n"
            
            for i, article in enumerate(articles[:8], 1):
                category = article['classified_category'] or article['category'] or 'General'
                sentiment_emoji = "😊" if article['sentiment_score'] and article['sentiment_score'] > 0.1 else "😔" if article['sentiment_score'] and article['sentiment_score'] < -0.1 else "😐"
                response += f"{i}. **[{category.upper()}]** {article['title']} {sentiment_emoji}\n"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in headlines response: {e}")
            return "I encountered an error while fetching headlines."
    
    def _generate_category_response(self, category: str) -> str:
        """Generate category-specific response"""
        try:
            articles = self.analyzer.get_articles(category=category, limit=10)
            
            if not articles:
                return f"I don't have any recent {category} news to show you."
            
            # Analyze category sentiment
            sentiment_scores = [a['sentiment_score'] for a in articles if a['sentiment_score'] is not None]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            sentiment_desc = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
            
            response = f"📂 **{category.title()} News Summary:**\n\n"
            response += f"Found {len(articles)} recent {category} articles with a {sentiment_desc} sentiment.\n\n"
            response += "🔥 **Top Headlines:**\n"
            
            for i, article in enumerate(articles[:5], 1):
                response += f"{i}. {article['title']}\n"
            
            if len(articles) > 5:
                response += f"\n... and {len(articles) - 5} more articles in this category."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in category response: {e}")
            return f"I encountered an error while fetching {category} news."
    
    def _generate_help_response(self) -> str:
        """Generate help response"""
        return """🤖 **AI News Assistant Help**

I can help you with:

📊 **News Analysis:**
• "Summarize the latest news"
• "What's happening in the news?"
• "Give me a news overview"

😊 **Sentiment Analysis:**
• "What's the current mood?"
• "How positive is the news today?"
• "Analyze the sentiment"

🏷️ **Topics & Categories:**
• "What are the trending topics?"
• "Show me technology news"
• "What's happening in politics?"

📰 **Headlines:**
• "Show me the latest headlines"
• "What are the top stories?"
• "Recent breaking news"

Just ask me naturally about current events, and I'll help you understand what's happening in the world!"""
    
    def _generate_general_response(self, message: str) -> str:
        """Generate general response for unclear intents"""
        return """I'm not sure exactly what you're looking for. Here are some things I can help you with:

• 📊 Get a summary of recent news
• 😊 Analyze the current sentiment/mood
• 🏷️ Show trending topics and categories  
• 📰 Display latest headlines
• 🔍 Find news about specific topics

Try asking something like:
- "What's the latest news?"
- "How positive is today's news?"
- "Show me technology headlines"
- "What's trending in politics?"

What would you like to know about current events?"""

if __name__ == "__main__":
    # Test the chatbot
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml')
    chatbot = ChatbotEngine(config_path)
    
    test_messages = [
        "Summarize the latest news",
        "What's the current mood?",
        "Show me technology news",
        "What are the trending topics?"
    ]
    
    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = chatbot.generate_response(msg)
        print(f"Bot: {response}")
        print("-" * 50)