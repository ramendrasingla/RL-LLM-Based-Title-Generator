import os
from dotenv import load_dotenv
from pathlib import Path
from transformers import GenerationConfig
from trl.core import LengthSampler

# Load environment variables from .env file
load_dotenv()

# Constants
API_KEY = os.getenv('GNEWS_API_KEY', "")
MAX_ARTICLES_PER_REQUEST = int(os.getenv('MAX_ARTICLES_PER_REQUEST', 25))
MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', 1))

RAW_DATA_PATH = Path('./data/raw/')
PREPROCESSED_DATA_PATH = Path('./data/preprocessed/')

# Create directories if they don't exist
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

DATA_INGESTION_OUTPUT_FILE = RAW_DATA_PATH / 'all_articles.json'
DATA_INGESTION_PIPELINE_LOG_FILE = RAW_DATA_PATH / 'pipeline_log.json'


# Define themes and their corresponding keywords
themes_keywords = {
    "Artificial Intelligence": ["AI", "Artificial Intelligence", "Machine Learning", "Deep Learning", "ChatGPT"],
    "Healthcare": ["Telemedicine", "Healthcare Technology", "Digital Health", "Medical AI", "COVID-19 vaccine"],
    "Climate Change": ["Climate Change", "Global Warming", "Carbon Emissions", "Environmental Policy"],
    "Politics": ["Election 2024", "US Politics", "Political Campaign", "Global Politics", "Policy Reform"],
    "Economy": ["Inflation", "Economic Policy", "Recession", "Stock Market", "Global Trade"],
    "Technology": ["Tech Innovations", "5G", "Blockchain", "Quantum Computing", "Cybersecurity"],
    "Entertainment": ["Movies", "Hollywood", "Streaming Platforms", "Oscars", "Celebrity News"],
    "Sports": ["Football", "Olympics", "FIFA World Cup", "Tennis", "NBA"],
    "Energy": ["Renewable Energy", "Electric Vehicles", "Oil Prices", "Solar Power", "Wind Energy"],
    "Education": ["Online Learning", "EdTech", "Remote Education", "STEM Education", "Higher Education"]
}

# Model Training Utils
SAMPLE = 4
LEARNING_RATE=1.41e-5
MAX_PPO_EPOCHS=10
MINI_BATCH_SIZE=4
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=16
MAX_LENGTH=512

output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)
MAX_NEW_TOKENS = output_length_sampler()

# MAX_TOKENS = 450 # For splitting into chunks
LORA_RANK_DIMS = 32
generation_kwargs = {
                     'max_new_tokens': MAX_NEW_TOKENS,
                     'top_k':50,
                     'top_p':0.85,
                     'temperature': 0.7,
                     'do_sample': True}

generation_config = GenerationConfig(**generation_kwargs)