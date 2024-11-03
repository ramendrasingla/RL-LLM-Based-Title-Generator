import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

from utils.data_prep.data_ingestion import data_ingestion_pipeline
from utils.data_prep.data_preprocessing import preprocess_text
from utils.data_prep.data_storage import save_dataframe
from utils.common.helper_funcs import setup_logging
from utils.common.constants import DATA_INGESTION_OUTPUT_FILE, PREPROCESSED_DATA_PATH

# Setup Logging
global logger
logger = setup_logging()

def parallelize_dataframe(df, func, num_partitions):
    # Split the dataframe into smaller chunks
    df_split = np.array_split(df, num_partitions)
    
    # Create a pool of workers
    with Pool(num_partitions) as pool:
        # Apply the function to each chunk in parallel
        df = pd.concat(pool.map(func, df_split))
    
    return df

def preprocess_text_columns(df):
    # Apply the preprocess_text function to the 'title' and 'content' columns
    df['title'] = df['title'].apply(preprocess_text)
    df['content'] = df['content'].apply(preprocess_text)
    return df

def prepare_dataset(load_type='init', run_data_ingest=True, splits=[0.2, 0.25]):
    if load_type == 'init' and run_data_ingest:
        logger.info("Starting initial data ingestion.")
        data_ingestion_pipeline(logger=logger, is_init_load=True)
        logger.info("Data ingestion completed for initial load.")
    elif run_data_ingest:
        logger.info("Starting regular data ingestion.")
        data_ingestion_pipeline(logger=logger, is_init_load=False)
        logger.info("Data ingestion completed for regular load.")
    
    logger.info("Loading and preprocessing data.")
    df = pd.read_json(DATA_INGESTION_OUTPUT_FILE)
    
    # Parallelize the preprocessing of text columns
    num_partitions = cpu_count()
    df = parallelize_dataframe(df, preprocess_text_columns, num_partitions)
    
    preprocessed_df = df[['title', 'content', 'theme']].drop_duplicates()
    logger.info("Data preprocessing completed.")

    # Train, Test Split
    logger.info("Splitting data into train, test, and validation sets.")
    train_val_df, test_df = train_test_split(preprocessed_df, test_size=splits[0], stratify=preprocessed_df['theme'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=splits[1], stratify=train_val_df['theme'], random_state=42)
    logger.info("Data splitting completed.")

    # Saving Processed dataframes
    save_type = "overwrite" if load_type == 'init' else "append"
    logger.info(f"Saving processed dataframes with save_type: {save_type}.")
    
    save_dataframe(logger, train_df, PREPROCESSED_DATA_PATH / "train.json", save_type)
    logger.info("Train data saved successfully.")
    
    save_dataframe(logger, test_df, PREPROCESSED_DATA_PATH / "test.json", save_type)
    logger.info("Test data saved successfully.")
    
    save_dataframe(logger, val_df, PREPROCESSED_DATA_PATH / "val.json", save_type)
    logger.info("Validation data saved successfully.")
    
    logger.info("Dataset preparation completed successfully.")



if __name__ == "__main__":


    # Ask the user if they want to run the initial load or delta load
    load_type = input("What kind of load type do you want to run(init/delta)? ").strip().lower()
    logger.info(f"Running {load_type} load...")
    
    run_data_ingest = input("Do we want to run data ingestion(y/n)? ").strip().lower()
    prepare_dataset(load_type= load_type, run_data_ingest= (run_data_ingest == 'y'))
