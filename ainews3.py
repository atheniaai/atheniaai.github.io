import requests
from datetime import datetime
import yaml
from typing import List, Dict
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import sys

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

def get_sentiment(text: str) -> str:
    """
    Analyze the sentiment of text and return a simple rating
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def is_relevant_article(title: str, description: str, topic: str) -> bool:
    """
    Check if the article is relevant to AI/ML topics
    """
    content = f"{title} {description}".lower()
    
    # TESTING MODE: Minimal filtering
    # Comment these basic exclusions for pure testing
    #basic_exclusions = {
    #    'podcast', 'song', 'music', 'movie'
    #}
    #if any(keyword in content for keyword in basic_exclusions):
    #    return False
        
    # For testing, return True to see all articles
    #return True
    
    
    # PRODUCTION MODE: Uncomment everything below for production
    
    # Core AI-specific keywords - must have at least one
    core_keywords = {
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'large language model', 'generative ai',
        'chatgpt', 'gpt-4', 'claude', 'gemini', 'llama', 'ai model',
        'computer vision', 'natural language processing', 'nlp',
        'openai', 'anthropic', 'deepmind', 'transformer'
    }
    
    # Technical keywords - must have at least one
    technical_keywords = {
        'algorithm', 'neural', 'model', 'training', 'inference',
        'parameter', 'fine-tuning', 'foundation model', 
        'pytorch', 'tensorflow', 'gpu', 'processor',
        'machine intelligence', 'cognitive computing'
    }
    
    # Exclude articles with these terms
    exclusion_keywords = {
        'trump', 'election', 'campaign', 'stock', 'market',
        'celebrity', 'movie', 'music', 'song', 'sport', 'weather',
        'podcast', 'skill gap', 'reskilling', 'upskilling',
        'phone launch', 'smartphone', 'mobile phone', 'tablet'
    }
    
    # Check for exclusion keywords first
    if any(keyword in content for keyword in exclusion_keywords):
        return False
    
    # Count occurrences of core and technical keywords
    core_count = sum(1 for keyword in core_keywords if keyword in content)
    technical_count = sum(1 for keyword in technical_keywords if keyword in content)
    
    # Accept if:
    # 1. Has at least one core keyword and one technical keyword
    # 2. Or has multiple core keywords
    return (core_count >= 1 and technical_count >= 1) or core_count >= 2
    

def fetch_ai_news(api_key: str) -> List[Dict]:
    """
    Fetch AI-related news using NewsAPI
    """
    base_url = "https://newsapi.org/v2/everything"
    request_count = 0
    
    # TESTING MODE: Single topic
    topics = ['artificial intelligence']
    
    """
    # PRODUCTION MODE: Uncomment below for all topics
    topics = [
        'artificial intelligence research',
        'machine learning technology',
        'large language model',
        'ChatGPT development',
        'OpenAI research',
        'Claude Anthropic',
        'NVIDIA AI computing',
        'deep learning neural',
        'generative AI model',
        'AI technology innovation',
        'AI research breakthrough',
        'neural network development',
        'foundation model AI',
        'transformer model AI',
        'LLM development'
    ]
    """
    
    # TESTING MODE: Basic exclusions
    exclusion_terms = 'NOT (podcast OR music OR movie)'
    
    """
    # PRODUCTION MODE: Uncomment for full exclusions
    exclusion_terms = (
        'NOT (politics OR election OR trump OR stock OR market OR '
        'celebrity OR movie OR music OR song OR sport OR weather OR '
        'podcast OR "skill gap" OR phone OR smartphone OR career OR '
        'education OR "job market")'
    )
    """
    
    all_articles = []
    print("\nFetching news articles...")
    
    for topic in topics:
        params = {
            'q': f'"{topic}" {exclusion_terms}',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,
            'apiKey': api_key
        }
        
        try:
            print(f"Making API request for topic: {topic}")
            request_count += 1
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            print(f"Found {len(articles)} initial articles")
            
            filtered_count = 0
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                
                if not title or not description:
                    continue
                
                if not is_relevant_article(title, description, topic):
                    continue
                
                filtered_count += 1
                description = BeautifulSoup(description, 'html.parser').get_text()
                sentiment = get_sentiment(f"{title} {description}")
                
                article_data = {
                    'title': title,
                    'preview': description,
                    'fullContent': article.get('content', description),
                    'image': article.get('urlToImage', ''),
                    'sourcelink': article.get('url'),
                    'topic': topic,
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name'),
                    'sentiment': sentiment
                }
                
                if not any(a['sourcelink'] == article_data['sourcelink'] for a in all_articles):
                    all_articles.append(article_data)
            
            print(f"Kept {filtered_count} articles after filtering")
            print(f"API requests made this session: {request_count}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {str(e)}")
            continue
    
    total_articles = len(all_articles)
    print(f"\nTotal unique articles after filtering and deduplication: {total_articles}")
    
    sorted_articles = sorted(all_articles, key=lambda x: x.get('publishedAt', ''), reverse=True)
    for i, article in enumerate(sorted_articles, 1):
        article['id'] = i
    
    return sorted_articles

def save_as_yaml(articles: List[Dict]) -> str:
    """
    Save articles in YAML format under news/MMDDYYYY/ainews.yaml
    """
    current_date = datetime.now()
    date_folder = current_date.strftime('%m%d%Y')
    directory = os.path.join('news', date_folder)
    filepath = os.path.join(directory, 'ainews.yaml')
    
    os.makedirs(directory, exist_ok=True)
    
    if os.path.exists(filepath):
        print(f"\nUpdating existing file at {filepath} with latest news data...")
    
    yaml_data = {'news': articles}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False, width=1000)
    
    return filepath

def main():
    
    try:
        articles = fetch_ai_news(API_KEY)
        
        if not articles:
            print("No articles found matching the criteria!")
            sys.exit(1)
            
        filepath = save_as_yaml(articles)
        
        print(f"News report saved to {filepath}")
        print(f"\nTotal unique articles found: {len(articles)}")
        
        topic_counts = {}
        for article in articles:
            topic = article['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        print("\nArticles per topic:")
        for topic, count in sorted(topic_counts.items()):
            print(f"{topic}: {count}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
