import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, GenerationConfig
from torch.utils.data import DataLoader, TensorDataset

from utils.constants import TRAIN_BATCH_SIZE, MAX_NEW_TOKENS

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_datasets(train_path, val_path, test_path, sample=None):
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load JSON files into Pandas DataFrames with NDJSON format
    try:
        train_df = pd.read_json(train_path, orient="records", lines=True)
        val_df = pd.read_json(val_path, orient="records", lines=True)
        test_df = pd.read_json(test_path, orient="records", lines=True)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        raise

    # Sample if sample size is specified, and reset the index
    if sample is not None:
        train_df = train_df.sample(n=sample).reset_index(drop=True)
        val_df = val_df.sample(n=sample).reset_index(drop=True)
        test_df = test_df.sample(n=sample).reset_index(drop=True)

    train_features = torch.tensor(train_df['content'].tolist())
    train_labels = torch.tensor(train_df['title'].tolist())

    val_features = torch.tensor(val_df['content'].tolist())
    val_labels = torch.tensor(val_df['title'].tolist())

    test_features = torch.tensor(test_df['content'].tolist())
    test_labels = torch.tensor(test_df['title'].tolist())

    # Create TensorDataset for train, validation, and test
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    # Combine into a single dataset dictionary
    dataset = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }

    return dataset

def build_dataset(tokenizer, sample = None):

    dataset = load_datasets(train_path = "./data/preprocessed/train.json",
                            test_path = "./data/preprocessed/test.json",
                            val_path = "./data/preprocessed/val.json",
                            sample = sample)

    def tokenize(dataset):
        
        # Wrap each article with the instruction.
        prompt = f"""
                What would be an appropriate title for the article?

                {dataset["content"]}

                Title:
                """
        dataset["input_ids"] = tokenizer.encode(prompt)
        
        # This must be called "query", which is a requirement of our PPO library.
        dataset["query"] = tokenizer.decode(dataset["input_ids"])
        return dataset
    
    # Tokenize each article
    splits = ['train', 'validation', 'test']
    for split in splits:
        
        dataset[split] = dataset[split].map(tokenize, batched=False)
        dataset[split].set_format(type="torch")

    return dataset


def load_reward_model(reward_model_name = "roberta-base-openai-detector"):


    reward_pipe = pipeline(reward_model_name + "-analysis", 
                            model=reward_model_name, 
                            device=device)
    reward_logits_kwargs = {
        "top_k": None, # Return all scores.
        "function_to_apply": "none", # Set to "none" to retrieve raw logits.
        "batch_size": TRAIN_BATCH_SIZE
    }

    return reward_pipe, reward_logits_kwargs


def get_latest_model_path(output_dir):
    # Function to get the latest model path from the output directory
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints:
        return max(checkpoints, key=os.path.getctime)
    

def trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"""
            \ntrainable model parameters: {trainable_model_params}
            \nall model parameters: {all_model_params}
            \npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%
            """

def probability_of_human_generated(input_text, evaluator_name = "roberta-base-openai-detector"):

    evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_name, device_map=device)
    evaluator_model = AutoModelForSequenceClassification.from_pretrained(evaluator_name, device_map=device)

    text_input_ids = evaluator_tokenizer(input_text, return_tensors="pt").input_ids
    logits = evaluator_model(input_ids=text_input_ids).logits

    probabilities = logits.softmax(dim=-1).tolist()[0]

    return probabilities[1]

def evaluate_humanity(model,
                      tokenizer, 
                      dataset, 
                      num_samples = None):
    
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model (trl model): Model to be evaluated.
    - evaluator (evaluate modules humanity metrics): Humanity evaluator.
    - tokenizer (transformers tokenizer): Tokenizer to be used.
    - dataset (dataset): Input dataset for the evaluation.
    - num_samples (int): Maximum number of samples for the evaluation.
        
    Returns:
    tuple: A tuple containing two numpy.float64 values:
    - mean (numpy.float64): Mean of the samples toxicity.
    - std (numpy.float64): Standard deviation of the samples toxicity.
    """

    humanity_scores = []
    for idx, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]

        if num_samples:
            if idx > num_samples:
                break
            
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        
        generation_config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS,
                                             top_k=0.0,
                                             top_p=1.0,
                                             do_sample=True)

        response_token_ids = model.generate(input_ids=input_ids,
                                            generation_config=generation_config)
        
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        
        humanity_score = probability_of_human_generated(generated_text)

        humanity_scores.extend(humanity_score)

    # Compute mean & std using np.
    mean = np.mean(humanity_scores)
    std = np.std(humanity_scores)
        
    return mean, std

def ppo_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)