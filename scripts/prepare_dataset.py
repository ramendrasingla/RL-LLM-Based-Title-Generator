import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from the .env file
load_dotenv(".env")

# Your News API key
API_KEY = os.getenv('API_KEY')

BASE_PATH = './data/raw'

# Function to fetch articles
def fetch_articles(from_date, to_date, page=1):
    url = (f'https://newsapi.org/v2/everything?'
           'domains=medium.com,techcrunch.com&'
           f'from={from_date}&'  # Start date
           f'to={to_date}&'      # End date
           f'page={page}&'       # Pagination
           'language=en&'
           'apiKey=' + API_KEY)

    response = requests.get(url)
    data = response.json()
    
    if data.get('status') == 'ok':
        return data['articles']
    else:
        print("Error fetching data:", data.get('message'))
        return []

# Function to save articles to CSV
def save_to_csv(articles, month_year):
    df = pd.DataFrame(articles)
    
    # Create directory if it doesn't exist
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    
    # Save to CSV
    csv_file_path = f'{BASE_PATH}/articles_{month_year}.csv'
    df.to_csv(csv_file_path, index=False, mode='a', header=not os.path.isfile(csv_file_path))
    print(f'Saved articles to {csv_file_path}')

# Initialize the last processed date
last_processed_date = None

# Load the last processed date from a file if it exists
if os.path.exists('last_processed_date.txt'):
    with open('last_processed_date.txt', 'r') as f:
        last_processed_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')

# Start from a default date if no last processed date exists
if last_processed_date is None:
    last_processed_date = datetime(2017, 1, 1)

# Loop through months until the present
end_date = datetime.now()

while last_processed_date < end_date:
    # Define the date range for the current month
    start_of_month = last_processed_date.replace(day=1)
    if last_processed_date.month == 12:
        end_of_month = start_of_month.replace(year=start_of_month.year + 1, month=1)
    else:
        end_of_month = start_of_month.replace(month=start_of_month.month + 1)

    # Format dates for API
    from_date = start_of_month.strftime('%Y-%m-%d')
    to_date = end_of_month.strftime('%Y-%m-%d')

    # Fetch articles for the current month
    page = 1
    all_articles = []

    while True:
        articles = fetch_articles(from_date, to_date, page)
        
        if not articles:
            break  # Stop if no articles are returned

        all_articles.extend(articles)
        page += 1

    # Partition articles by month-year
    if all_articles:
        month_year = start_of_month.strftime('%Y-%m')
        save_to_csv(all_articles, month_year)
        
        # Update last processed date to the end of the current month
        last_processed_date = end_of_month

# Save the last processed date
with open('last_processed_date.txt', 'w') as f:
    f.write(last_processed_date.strftime('%Y-%m-%d'))

print("Article fetching and saving completed.")