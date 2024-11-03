import os
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from utils.data_prep.data_storage import (load_oldest_article_dates, save_articles_to_json, save_oldest_article_dates)
from utils.common.constants import (API_KEY, MAX_ARTICLES_PER_REQUEST, MAX_ITERATIONS,
                            DATA_INGESTION_OUTPUT_FILE, DATA_INGESTION_PIPELINE_LOG_FILE, 
                            themes_keywords)

def fetch_articles(logger, search_keyword, oldest_date=None):
    """Fetch articles from GNews API for a specific keyword, using oldest_date for the 'to' parameter."""
    all_articles = []

    if isinstance(oldest_date, str):
        oldest_date = datetime.fromisoformat(oldest_date)

        # Update to 10 days before the oldest date
        updated_oldest_date = oldest_date - timedelta(days=10)
    
    else:
        updated_oldest_date = None

    params = {
        'q': search_keyword,
        'lang': 'en',
        'sortby': 'relevance',
        'token': API_KEY,
        'to': updated_oldest_date.isoformat() + 'Z' if updated_oldest_date else None,
        'max': MAX_ARTICLES_PER_REQUEST,
        'expand': 'content'
    }

    logger.info(f"Sending request for keyword: {search_keyword}, params: {params}")
    response = requests.get('https://gnews.io/api/v4/search', params=params)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        logger.info(f"Received {len(articles)} articles for keyword: {search_keyword}")

        if not articles:
            logger.info(f"No articles found for keyword: {search_keyword}")

        all_articles.extend([
            {
                'keyword': search_keyword,
                'title': article['title'],
                'content': article['content'],
                'published_date': datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00")).isoformat(),
                'id': article['url'] # Assuming URL can be used as a unique identifier
            }
            
            for article in articles
        ])
    else:
        logger.error(f"Failed to fetch articles: {response.status_code} - {response.text}")

    return all_articles


def data_ingestion_pipeline(logger, is_init_load):
    """Data preparation pipeline for fetching and storing articles based on keywords."""
    

    if is_init_load:
        # Overwrite the existing articles JSON if it's an initial load
        if os.path.exists(DATA_INGESTION_OUTPUT_FILE):
            os.remove(DATA_INGESTION_OUTPUT_FILE)
        if os.path.exists(DATA_INGESTION_PIPELINE_LOG_FILE):
            os.remove(DATA_INGESTION_PIPELINE_LOG_FILE)
        logger.info("Initial load: Overwriting the existing articles file.")
    
    for itr in tqdm(range(MAX_ITERATIONS)):
        # Load the oldest dates for delta load to avoid duplicates
        oldest_article_dates = load_oldest_article_dates(logger) if ((not is_init_load) or (itr != 0))  else {}

        # Debugging: Log the structure of oldest_article_dates
        logger.debug(f"Oldest article dates loaded: {oldest_article_dates}")
        logger.debug(oldest_article_dates.keys())
        
        for theme, keywords in themes_keywords.items():
            for keyword in keywords:
                # Fetch the oldest date, checking for its presence
                if (not is_init_load) or (itr != 0):
                    oldest_date = oldest_article_dates.get(keyword)
                else:
                    oldest_date = None

                logger.info(f"Fetching articles for keyword: '{keyword}'")

                
                # Debugging: Log the keyword and its corresponding oldest date
                logger.debug(f"Keyword: {keyword}, Oldest date fetched: {oldest_date}")

                # If we're not doing an initial load and oldest_date is None, raise an error
                # if((not is_init_load) or (itr != 0)) and not oldest_date:
                #     raise ValueError(f"Error: Oldest date for keyword '{keyword}' is None.")

                # Log the oldest date based on the keyword
                if oldest_date is not None:
                    logger.info(f"Oldest date for keyword '{keyword}': {oldest_date}")

                logger.info(f"Fetching articles for keyword: '{keyword}' with oldest date: {oldest_date}")
                fetched_articles = fetch_articles(logger, keyword, oldest_date)

                if fetched_articles:
                    logger.info(f"Fetched {len(fetched_articles)} articles for keyword: {keyword}")
                    save_articles_to_json(fetched_articles, theme)  # Adjust as necessary

                    try:
                        # Ensure all articles have a valid 'published_date' and handle errors
                        dates = [
                            datetime.fromisoformat(article['published_date'])
                            for article in fetched_articles
                            if 'published_date' in article and article['published_date']  # Check if the key exists and is not empty
                        ]
                        
                        if dates:
                            new_oldest_date = min(dates).isoformat()
                            # Update the oldest date in the dictionary
                            oldest_article_dates[keyword] = new_oldest_date
                            # Save the updated oldest article dates
                            save_oldest_article_dates(oldest_article_dates)
                            logger.info(f"Updated oldest date for keyword '{keyword}': {new_oldest_date}")
                        else:
                            logger.warning(f"No valid published dates found for keyword: {keyword}")

                    except ValueError as e:
                        logger.error(f"Error parsing dates for keyword '{keyword}': {e}")

            
