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
    Check if the article is relevant to AI/ML topics with balanced filtering
    """
    content = f"{title} {description}".lower()
    
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
    Fetch AI-related news using NewsAPI (Free tier version)
    """
    base_url = "https://newsapi.org/v2/everything"
    
    # Expanded list of topics to get more coverage within free tier limits
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
    
    exclusion_terms = (
        'NOT (politics OR election OR trump OR stock OR market OR '
        'celebrity OR movie OR music OR song OR sport OR weather OR '
        'podcast OR "skill gap" OR phone OR smartphone OR career OR '
        'education OR "job market")'
    )
    
    all_articles = []
    print("\nFetching news articles...")
    
    for topic in topics:
        params = {
            'q': f'"{topic}" {exclusion_terms}',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100,  # Maximum allowed in free tier
            'apiKey': api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            print(f"Found {len(articles)} initial articles for topic: {topic}")
            
            filtered_count = 0
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Skip if title or description is missing
                if not title or not description:
                    continue
                
                # Apply relevance checking
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
                
                # Only add if we haven't seen this URL before
                if not any(a['sourcelink'] == article_data['sourcelink'] for a in all_articles):
                    all_articles.append(article_data)
            
            print(f"Kept {filtered_count} articles after filtering for topic: {topic}")
            print(f"Current total unique articles: {len(all_articles)}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for topic '{topic}': {str(e)}")
            continue
    
    total_articles = len(all_articles)
    print(f"\nTotal unique articles after filtering and deduplication: {total_articles}")
    
    # Sort articles by published date and add numeric IDs
    sorted_articles = sorted(all_articles, key=lambda x: x.get('publishedAt', ''), reverse=True)
    for i, article in enumerate(sorted_articles, 1):
        article['id'] = i
    
    return sorted_articles
 
    total_articles = len(all_articles)
    print(f"\nTotal unique articles after filtering and deduplication: {total_articles}")
    
    if total_articles < 30:
        print("Warning: Found fewer than 30 articles. Adjusting filtering criteria...")
        # You might want to implement alternative logic here
    
    # Sort articles by published date and add numeric IDs
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
