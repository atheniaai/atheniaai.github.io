import requests
from datetime import datetime, timedelta
import yaml
from typing import List, Dict
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import sys
import shutil
from openai import OpenAI
from openai import AzureOpenAI
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_and_save_image_dalle3(self, prompt: str, save_path: str) -> str:
    """Generate image using DALL-E with detailed error logging."""
    try:
        self.logger.info(f"Attempting to generate image with prompt: {prompt}")
        # List available models
        models = self.azure_OAI_client.models.list()
        self.logger.debug(f"Available OpenAI models: {models}")
        # Try DALL-E 3 first, fall back to DALL-E 2 if needed
        try:
            response = self.azure_OAI_client.images.generate(
                model="dall-e-3",
                prompt=f"o: {prompt}. A dynamic and visually engaging image suitable for a news media website. The composition is clean and professional, with vibrant and realistic details, designed to evoke interest and convey a sense of importance or immediacy. No text or lettering should be visible on the image.",
                n=1,
                size="1024x1024"
            )
        except Exception as e:
            self.logger.warning(f"DALL-E 3 failed, trying DALL-E 2: {str(e)}")
            response = self.client.images.generate(
                model="dall-e-2",
                prompt=f"create an image: {prompt}. Style: Modern, clean.",
                n=1,
                size="1024x1024"
            )
        
        self.logger.debug(f"OpenAI image generation response: {response}")
        image_url = response.data[0].url
        self.logger.info(f"Successfully generated image URL: {image_url}")
        
        # Download image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            # Process image with PIL for compression
            from PIL import Image
            from io import BytesIO
            
            # Open image from response content
            img = Image.open(BytesIO(img_response.content))
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gc.collect()
            
            # Compress and save with target size between 250-300KB
            quality = 95
            while True:
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                size_kb = buffer.tell() / 1024
                
                if size_kb < 250:
                    quality = min(quality + 5, 95)  # Increase quality if too small
                elif size_kb > 300:
                    quality = max(quality - 5, 20)  # Decrease quality if too large
                else:
                    # Size is in target range, save the file
                    img.save(save_path, 'JPEG', quality=quality, optimize=True)
                    self.logger.info(f"Successfully saved compressed image ({size_kb:.1f}KB) to: {save_path}")
                    buffer.close()
                    del buffer
                    break
                buffer.close()
                del buffer
                gc.collect()
                if quality == 20 or quality == 95:  # Prevent infinite loop
                    img.save(save_path, 'JPEG', quality=quality, optimize=True)
                    final_size = os.path.getsize(save_path) / 1024
                    self.logger.info(f"Saved image with best possible compression ({final_size:.1f}KB) to: {save_path}")
                    break
            img.close()
            del img
            gc.collect()
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            self.logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
            return save_path
        else:
            self.logger.error(f"Failed to download image. Status code: {img_response.status_code}")
            return ""
    except Exception as e:
        self.logger.error(f"Error in image generation: {str(e)}")
        self.logger.error(f"Error details: {e}")
        return ""

def copy_to_next_day():
    """
    Copy the entire contents of current date directory to next date directory.
    Both directories are relative to the workflow environment.
    """
    # Get current and next day dates
    current_date = datetime.now()
    next_date = current_date + timedelta(days=1)
    
    # Format dates as MMDDYYYY
    current_dir_name = current_date.strftime('%m%d%Y')
    next_dir_name = next_date.strftime('%m%d%Y')
    
    # Set up paths relative to workflow directory
    base_path = os.path.join('news')
    current_dir = os.path.join(base_path, current_dir_name)
    next_dir = os.path.join(base_path, next_dir_name)
    
    # Create the next day directory (including parent directories if they don't exist)
    os.makedirs(next_dir, exist_ok=True)
    
    try:
        # Copy everything from current to next day directory
        for item in os.listdir(current_dir):
            source = os.path.join(current_dir, item)
            destination = os.path.join(next_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
                
        print(f"Successfully copied contents from {current_dir} to {next_dir}")
    except Exception as e:
        print(f"Error copying directory: {str(e)}")
        raise

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
    return (core_count >= 2 and technical_count >= 1) or core_count >= 2

def get_news_directory() -> str:
    """Get the news directory path for current date"""
    current_date = datetime.now().strftime('%m%d%Y')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'news', current_date)

def get_image_path(news_dir: str, article_id: int) -> tuple:
    """
    Get the image directory and path
    Returns tuple of (save_path, relative_path)
    """
    image_dir = os.path.join(news_dir, 'images', str(article_id))
    save_path = os.path.join(image_dir, 'article_image.png')
    # Relative path for YAML
    relative_path = os.path.join('news', os.path.basename(news_dir), 'images', str(article_id), 'article_image.png')
    return save_path, relative_path

def save_as_yaml(articles: List[Dict]) -> str:
    """
    Save articles in YAML format under news/MMDDYYYY/ainews.yaml
    """
    news_dir = get_news_directory()
    filepath = os.path.join(news_dir, 'ainews.yaml')
    
    os.makedirs(news_dir, exist_ok=True)
    
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
    def __init__(self, openai_api_key: str, azure_oai_key: str, azure_oai_url :str, azure_oai_version: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.azure_OAI_client = AzureOpenAI(api_key=azure_oai_key, api_version=azure_oai_version, azure_endpoint=azure_oai_url)
        self.logger = logger

    def generate_preview(self, title: str) -> str:
        """Generate 5-word preview using GPT-4"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
            models = self.azure_OAI_client.models.list()
            self.logger.debug(f"Available OpenAI models: {models}")
            # Try DALL-E 3 first, fall back to DALL-E 2 if needed
            try:
                response = self.azure_OAI_client.images.generate(
                    model="dall-e-3",
                    prompt=f"o: {prompt}. A dynamic and visually engaging image suitable for a news media website. The composition is clean and professional, with vibrant and realistic details, designed to evoke interest and convey a sense of importance or immediacy. No text or lettering should be visible on the image.",
                    n=1,
                    size="1024x1024"
                )
            except Exception as e:
                self.logger.warning(f"DALL-E 3 failed, trying DALL-E 2: {str(e)}")
                response = self.client.images.generate(
                    model="dall-e-2",
                    prompt=f"create an image: {prompt}. Style: Modern, clean.",
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
			
    def generate_and_save_image_dalle3(self, prompt: str, save_path: str) -> str:
        """Generate image using DALL-E with detailed error logging."""
        try:
            self.logger.info(f"Attempting to generate image with prompt: {prompt}")
            # List available models
            models = self.azure_OAI_client.models.list()
            self.logger.debug(f"Available OpenAI models: {models}")
            # Try DALL-E 3 first, fall back to DALL-E 2 if needed
            try:
                response = self.azure_OAI_client.images.generate(
                    model="dall-e-3",
                    prompt=f"o: {prompt}. A dynamic and visually engaging image suitable for a news media website. The composition is clean and professional, with vibrant and realistic details, designed to evoke interest and convey a sense of importance or immediacy. No text or lettering should be visible on the image.",
                    n=1,
                    size="1024x1024"
                )
            except Exception as e:
                self.logger.warning(f"DALL-E 3 failed, trying DALL-E 2: {str(e)}")
                response = self.client.images.generate(
                    model="dall-e-2",
                    prompt=f"create an image: {prompt}. Style: Modern, clean.",
                    n=1,
                    size="1024x1024"
                )
            
            self.logger.debug(f"OpenAI image generation response: {response}")
            image_url = response.data[0].url
            self.logger.info(f"Successfully generated image URL: {image_url}")
            
            # Download image
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # Process image with PIL for compression
                from PIL import Image
                from io import BytesIO
                
                # Open image from response content
                img = Image.open(BytesIO(img_response.content))
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Compress and save with target size between 250-300KB
                quality = 95
                while True:
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=quality, optimize=True)
                    size_kb = buffer.tell() / 1024
                    
                    if size_kb < 250:
                        quality = min(quality + 5, 95)  # Increase quality if too small
                    elif size_kb > 300:
                        quality = max(quality - 5, 20)  # Decrease quality if too large
                    else:
                        # Size is in target range, save the file
                        img.save(save_path, 'JPEG', quality=quality, optimize=True)
                        self.logger.info(f"Successfully saved compressed image ({size_kb:.1f}KB) to: {save_path}")
                        break
                    
                    if quality == 20 or quality == 95:  # Prevent infinite loop
                        img.save(save_path, 'JPEG', quality=quality, optimize=True)
                        final_size = os.path.getsize(save_path) / 1024
                        self.logger.info(f"Saved image with best possible compression ({final_size:.1f}KB) to: {save_path}")
                        break
                
                return save_path
            else:
                self.logger.error(f"Failed to download image. Status code: {img_response.status_code}")
                return ""
        except Exception as e:
            self.logger.error(f"Error in image generation: {str(e)}")
            self.logger.error(f"Error details: {e}")
            return ""

def fetch_ai_news(api_key: str, openai_api_key: str,  azure_oai_key: str, azure_oai_url :str, azure_oai_version: str, max_articles: int = 2) -> List[Dict]:
    """
    Fetch AI-related news using NewsAPI and process with OpenAI
    """
    base_url = "https://newsapi.org/v2/everything"
    request_count = 0
    news_processor = NewsProcessor(openai_api_key, azure_oai_key, azure_oai_url, azure_oai_version)
    
    # Get news directory for current date
    news_dir = get_news_directory()
    
    # TESTING MODE: Single topic
    #topics = ['GPT-4']
    #exclusion_terms = 'NOT (podcast OR music OR movie)'

    
    # PRODUCTION MODE: Uncomment below for all topics
    topics = [
        'GPT-4',
        'Claude AI',
        'artificial intelligence',
        'neural network development',
        'machine learning technology',
        'GPT-4',
        'NVIDIA AI computing',
        'LLM development',
        'deep learning neural',
        'Claude AI',
        'OpenAI research',
        'large language model',
        'ChatGPT development',
        'Claude Anthropic',
        'generative AI model',
        'AI technology innovation',
        'AI research breakthrough',
        'neural network development',
        'artificial intelligence research',
        'foundation model AI',
        'transformer model AI',
        'LLM development'
    ]
    """
    
    # TESTING MODE: Basic exclusions
    #exclusion_terms = 'NOT (podcast OR music OR movie)'
    
    """
    # PRODUCTION MODE: Uncomment for full exclusions
    exclusion_terms = (
        'NOT (politics OR election OR trump OR stock OR market OR '
        'celebrity OR movie OR music OR song OR sport OR weather OR '
        'podcast OR "skill gap" OR phone OR smartphone OR career OR '
        'education OR "job market")'
    )

    
    all_articles = []
    processed_count = 0
    print(f"\nFetching news articles (will process maximum {max_articles} articles)...")
    
    for topic in topics:
        if processed_count >= max_articles:
            print(f"Reached maximum article limit of {max_articles}")
            break
            
        params = {
            'q': f'"{topic}" {exclusion_terms}',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
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
                
                preview = news_processor.generate_preview(title)
                full_content = news_processor.summarize_content(description)
                
                article_data = {
                    'title': title,
                    'preview': preview,
                    'fullContent': full_content,
                    'img': article.get('urlToImage', ''),
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
    
    # Sort and add IDs, generate images
    sorted_articles = sorted(all_articles, key=lambda x: x.get('publishedAt', ''), reverse=True)
    for i, article in enumerate(sorted_articles, 1):
        article['id'] = i
        
        # Get image paths
        save_path, relative_path = get_image_path(news_dir, article['id'])
        
        # Ensure image directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Generate image using the summarized content as prompt
        if news_processor.generate_and_save_image_dalle3(article['fullContent'], save_path):
            article['image'] = relative_path
        else:
            article['image'] = ''
    
    return sorted_articles

def main():
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AZURE_OPENAI_DALLE_KEY = os.getenv('AZURE_OPENAI_DALLE_KEY')
    AZURE_OPENAI_DALLE_URL = os.getenv('AZURE_OPENAI_DALLE_URL')
    AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')
    if not NEWS_API_KEY or not OPENAI_API_KEY or not AZURE_OPENAI_DALLE_KEY or not AZURE_OPENAI_DALLE_URL:
        raise ValueError("API keys not found in environment variables")
    
    # Configure maximum articles to process
    MAX_ARTICLES = 20  # Change this value for testing
    
    try:
        articles = fetch_ai_news(
            api_key=NEWS_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            azure_oai_key=AZURE_OPENAI_DALLE_KEY,
            azure_oai_url=AZURE_OPENAI_DALLE_URL,
            azure_oai_version=AZURE_API_VERSION,
            max_articles=MAX_ARTICLES
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
    copy_to_next_day()
