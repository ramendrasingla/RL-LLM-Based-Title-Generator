import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_ingestion import data_ingestion_pipeline
from utils.data_preprocessing import preprocess_text
from utils.data_storage import save_dataframe
from utils.helper_funcs import setup_logging
from utils.constants import DATA_INGESTION_OUTPUT_FILE, PREPROCESSED_DATA_PATH

# Setup Logging
global logger
logger = setup_logging()

def prepare_dataset(load_type = 'init', run_data_ingest = True, splits = [0.2, 0.25]):

    if load_type == 'init' and run_data_ingest:
        # Data Ingestion
        data_ingestion_pipeline(logger=logger, is_init_load=True)
            
    elif run_data_ingest:
        # Data Ingestion
        data_ingestion_pipeline(logger=logger, is_init_load=False)
    
    # Data Preprocessing
    df = pd.read_json(DATA_INGESTION_OUTPUT_FILE)
    df['title'] = df['title'].apply(preprocess_text)
    df['content'] = df['content'].apply(preprocess_text)
    preprocessed_df = df[['title', 'content', 'theme']].drop_duplicates()

    # Train, Test Split
    train_val_df, test_df = train_test_split(preprocessed_df, test_size=splits[0], stratify=preprocessed_df['theme'], random_state=42)
    # Train, Validation Split
    train_df, val_df = train_test_split(train_val_df, test_size=splits[1], stratify=train_val_df['theme'], random_state=42)

    # Saving Processed dataframes
    if load_type == 'init':
        save_type = "overwrite"
    else:
        save_type = "append"

    save_dataframe(logger, train_df, PREPROCESSED_DATA_PATH / "train.json", save_type)
    save_dataframe(logger, test_df, PREPROCESSED_DATA_PATH / "test.json", save_type)
    save_dataframe(logger, val_df, PREPROCESSED_DATA_PATH / "val.json", save_type)


if __name__ == "__main__":


    # Ask the user if they want to run the initial load or delta load
    load_type = input("What kind of load type do you want to run(init/delta)? ").strip().lower()
    logger.info(f"Running {load_type} load...")
    if load_type == 'delta':
        # In case of delta load, ask if we want to run data ingestion or not
        run_data_ingest = input("Do we want to run data ingestion(y/n)? ").strip().lower()
        prepare_dataset(load_type= load_type, run_data_ingest= (run_data_ingest == 'y'))
    else:
        prepare_dataset(load_type= load_type, run_data_ingest= True)