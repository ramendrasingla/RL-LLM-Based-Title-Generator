import json
import os
import pandas as pd
from pathlib import Path
from utils.common.constants import DATA_INGESTION_OUTPUT_FILE, DATA_INGESTION_PIPELINE_LOG_FILE

def load_oldest_article_dates(logger):
    """Load the oldest published dates for each theme from a JSON file."""
    if not os.path.exists(DATA_INGESTION_PIPELINE_LOG_FILE):
        # Return an empty dictionary if the log file does not exist
        logger.info(f"No log file found, starting with an empty set of oldest article dates.")
        return {}

    try:
        with open(DATA_INGESTION_PIPELINE_LOG_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Handle cases where the file exists but contains invalid JSON
        logger.error(f"Log file {DATA_INGESTION_PIPELINE_LOG_FILE} contains invalid JSON.")
        return {}
    except Exception as e:
        logger.error(f"An error occurred while loading the log file: {e}")
        return {}

def save_oldest_article_dates(oldest_dates):
    """Save the oldest published dates for each theme to a JSON file."""
    with open(DATA_INGESTION_PIPELINE_LOG_FILE, 'w') as f:
        json.dump(oldest_dates, f, indent=4)

def save_articles_to_json(articles, theme):
    """Save fetched articles to a single JSON file, overwriting for init load or appending for delta load."""
    output_data = []

    # Add theme and theme_id to each article
    for article in articles:
        article.update({
            'theme': theme
        })
        output_data.append(article)
    
    if os.path.exists(DATA_INGESTION_OUTPUT_FILE):
        with open(DATA_INGESTION_OUTPUT_FILE, 'r') as f:
            existing_data = json.load(f)
            output_data = existing_data + output_data

    with open(DATA_INGESTION_OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)


def save_dataframe(logger, df, file_path, save_type="overwrite"):
    file_path = Path(file_path)
    
    if save_type == "overwrite":
        df.to_json(file_path, orient="records", lines=True)
        logger.info(f"Data overwritten to {file_path}")
    
    elif save_type == "append":
        # Check if file exists, append the data
        if file_path.exists():
            # Load existing data
            existing_df = pd.read_json(file_path, orient="records", lines=True)
            # Concatenate the new data
            combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
        else:
            # If the file doesn't exist, just save the new data
            combined_df = df
        
        # Save the combined data back
        combined_df.to_json(file_path, orient="records", lines=True)
        logger.info(f"Data appended to {file_path}")
    