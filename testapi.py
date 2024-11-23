import requests

def test_api_key(api_key: str):
    """
    Test if the API key has reached its daily limit
    """
    base_url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': 'artificial intelligence',
        'language': 'en',
        'pageSize': 1,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 429:
            print("Rate limit exceeded. Please try again tomorrow.")
            return False
            
        if response.status_code == 426:
            print("Free tier limit reached. Please try again tomorrow.")
            return False
            
        response_json = response.json()
        
        if response_json.get('status') == 'error':
            print(f"API Error: {response_json.get('message')}")
            return False
            
        print("API key is working and has available requests.")
        return True
        
    except Exception as e:
        print(f"Error testing API key: {str(e)}")
        return False

# Usage
API_KEY = "4e086be4d63e48559556648439510d34"
test_api_key(API_KEY)
