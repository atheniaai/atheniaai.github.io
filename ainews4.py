import requests
from datetime import datetime
import yaml
from typing import List, Dict
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import sys
import shutil
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_relevant_article(title: str, description: str, topic: str) -> bool:
    """
    Check if the article is relevant to AI/ML topics
    """
    content = f"{title} {description}".lower()

    # TESTING MODE: Minimal filtering
    # Comment these basic exclusions for pure testing
    basic_exclusions = {
        'podcast', 'song', 'music', 'movie'
    }
    if any(keyword in content for keyword in basic_exclusions):
        return False

    # For testing, return True to see all articles
    return True


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

# Initialize OpenAI client
class NewsProcessor:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.logger = logger

    def generate_preview(self, title: str) -> str:
        """Generate 5-word preview using GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Generate a compelling 5-word summary of the given title."},
                    {"role": "user", "content": f"Title: {title}"}
                ],
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating preview: {e}")
            return title[:50]  # Fallback to truncated title

    def summarize_content(self, content: str) -> str:
        """Summarize content to 100 words using GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Summarize the following text in exactly 100 words or less."},
                    {"role": "user", "content": content}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error summarizing content: {e}")
            return content[:500]  # Fallback to truncated content

    def generate_and_save_image(self, prompt: str, save_path: str) -> str:
        """Generate image using DALL-E with detailed error logging."""
        try:
            self.logger.info(f"Attempting to generate image with prompt: {prompt}")
            # List available models
            models = self.client.models.list()
            self.logger.debug(f"Available OpenAI models: {models}")
            # Try DALL-E 3 first, fall back to DALL-E 2 if needed
            try:
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=f"o: {prompt}. Style: Modern, clean,",
                    n=1,
                    size="1024x1024"
                )
            except Exception as e:
                self.logger.warning(f"DALL-E 3 failed, trying DALL-E 2: {str(e)}")
                response = self.client.images.generate(
                    model="dall-e-2",
                    prompt=f"create an image: {prompt}. Style: Modern, clean, financial.",
                    n=1,
                    size="1024x1024"
                )
            
            self.logger.debug(f"OpenAI image generation response: {response}")
            image_url = response.data[0].url
            self.logger.info(f"Successfully generated image URL: {image_url}")
            
            # Download and save image
            img_response = requests.get(image_url, stream=True)
            if img_response.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    img_response.raw.decode_content = True
                    shutil.copyfileobj(img_response.raw, f)
                self.logger.info(f"Successfully saved image to: {save_path}")
                return save_path
            else:
                self.logger.error(f"Failed to download image. Status code: {img_response.status_code}")
                return ""
        except Exception as e:
            self.logger.error(f"Error in image generation: {str(e)}")
            self.logger.error(f"Error details: {e}")
            return ""

def fetch_ai_news(api_key: str, openai_api_key: str, max_articles: int = 2) -> List[Dict]:
    """
    Fetch AI-related news using NewsAPI and process with OpenAI
    Args:
        api_key: NewsAPI key
        openai_api_key: OpenAI API key
        max_articles: Maximum number of articles to process (default: 2)
    """
    base_url = "https://newsapi.org/v2/everything"
    request_count = 0
    news_processor = NewsProcessor(openai_api_key)
    
    # TESTING MODE: Single topic
    topics = ['artificial intelligence']
    exclusion_terms = 'NOT (podcast OR music OR movie)'
    
    all_articles = []
    processed_count = 0
    print(f"\nFetching news articles (will process maximum {max_articles} articles)...")
    
    for topic in topics:
        # Break if we've reached the maximum articles
        if processed_count >= max_articles:
            print(f"Reached maximum article limit of {max_articles}")
            break
            
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
                # Break if we've reached the maximum articles
                if processed_count >= max_articles:
                    print(f"Reached maximum article limit of {max_articles}")
                    break
                    
                title = article.get('title', '')
                description = article.get('description', '')
                
                if not title or not description:
                    continue
                
                if not is_relevant_article(title, description, topic):
                    continue
                
                filtered_count += 1
                description = BeautifulSoup(description, 'html.parser').get_text()
                
                # Generate preview using GPT-4
                preview = news_processor.generate_preview(title)
                
                # Summarize content using GPT-4
                full_content = news_processor.summarize_content(description)
                
                article_data = {
                    'title': title,
                    'preview': preview,
                    'fullContent': full_content,
                    'image': article.get('urlToImage', ''),
                    'sourcelink': article.get('url'),
                    'topic': topic,
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name'),
                    'sentiment': get_sentiment(f"{title} {description}")
                }
                
                if not any(a['sourcelink'] == article_data['sourcelink'] for a in all_articles):
                    all_articles.append(article_data)
                    processed_count += 1
                    print(f"Processed article {processed_count} of {max_articles}")
            
            print(f"Kept {filtered_count} articles after filtering")
            print(f"API requests made this session: {request_count}")
                
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            continue
    
    # Sort and add IDs
    sorted_articles = sorted(all_articles, key=lambda x: x.get('publishedAt', ''), reverse=True)
    for i, article in enumerate(sorted_articles, 1):
        article['id'] = i
        
        # Generate and save image for each article
        image_dir = os.path.join('images', str(article['id']))
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, 'article_image.png')
        
        # Generate image using the summarized content as prompt
        news_processor.generate_and_save_image(article['fullContent'], image_path)
        article['generated_image'] = image_path
    
    return sorted_articles

def main():
    
    # Configure maximum articles to process
    # TESTING MODE: Set to small number (2-5) for testing
    # PRODUCTION MODE: Set to larger number or comment out to process all articles
    MAX_ARTICLES = 2  # Change this value for testing
    
    try:
        articles = fetch_ai_news(
            api_key=NEWS_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            max_articles=MAX_ARTICLES  # Pass the configuration
        )
        
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
