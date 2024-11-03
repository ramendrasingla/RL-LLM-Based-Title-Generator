import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict

from utils.common.constants import generation_config, CHECKPOINT_DIR_PATH, EVAL_BATCH_SIZE, MAX_LENGTH

# Setup device
device = torch.device('mps' if torch.mps.is_available() else ('gpu' if torch.gpu.is_available() else 'cpu'))

def load_datasets(train_path, val_path, test_path, sample=None, train_valid_together = True):
    # Load JSON files into Pandas DataFrames
    train_df = pd.read_json(train_path, orient="records", lines=True)
    val_df = pd.read_json(val_path, orient="records", lines=True)
    test_df = pd.read_json(test_path, orient="records", lines=True)

    # Optionally sample the datasets
    if sample is not None:
        train_df = train_df.sample(n=sample).reset_index(drop=True)
        val_df = val_df.sample(n=sample).reset_index(drop=True)
        test_df = test_df.sample(n=sample).reset_index(drop=True)

    if train_valid_together:

        train_df = pd.concat([train_df, val_df], axis=0)

        # Convert the DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Combine into a DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

    else:
        # Convert the DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Combine into a DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

    return dataset

def build_dataset(logger, tokenizer, sample=None):
    # Load the datasets as pandas DataFrames
    dataset = load_datasets(
        train_path="./data/preprocessed/train.json",
        val_path="./data/preprocessed/val.json",
        test_path="./data/preprocessed/test.json",
        sample=sample,
        #max_tokens=MAX_TOKENS
    )

    if dataset is None:
        logger.error("Failed to load datasets. Exiting build_dataset.")
        return None

    # Tokenize each sample and generate prompts for each chunk
    def tokenize(sample):
        # TODO: Pick prompt from constants
        # Define your prompt

        prompt = f"""
                    Write a catchy title that summarizes the primary message and unique insights from this text.
                    \n\nText:{sample["content"]}
                    \n\nGenerate a title:
                    """
        # Tokenize the prompt
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # Apply tokenization to the dataset
    dataset = dataset.map(tokenize, batched=False)

    # Set format to PyTorch tensors (if required for further processing)
    dataset.set_format(type="torch")

    return dataset
    

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

# Load evaluator model and tokenizer once
evaluator_name = "roberta-base-openai-detector"
evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_name)
evaluator_model = AutoModelForSequenceClassification.from_pretrained(evaluator_name).to(device)

# Function to get probability of human-generated text
def probability_of_human_generated(input_text):
    text_input_ids = evaluator_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    with torch.no_grad():  # Disable gradient calculation for inference
        logits = evaluator_model(input_ids=text_input_ids).logits
    probabilities_batch = logits.softmax(dim=-1).cpu().tolist()
    probabilities = [prob[1] for prob in probabilities_batch]
    return probabilities

def evaluate_humanity(model, tokenizer, dataset, num_samples=None, batch_size=EVAL_BATCH_SIZE):
    humanity_scores = []
    gen_texts = []

    # Ensure input texts are strings and flatten if needed
    input_texts = [str(sample["query"]) for sample in dataset[:num_samples]] if num_samples else [str(sample["query"]) for sample in dataset]

    # Process the dataset in batches with a progress bar
    for i in tqdm(range(0, len(input_texts), batch_size), desc="Processing batches"):
        batch_input_texts = input_texts[i:i + batch_size]

        try:
            # Tokenize input texts for the current batch with padding and truncation enabled
            tokenized_inputs = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            input_ids = tokenized_inputs["input_ids"].to(device)

            # Generate responses for the current batch
            with torch.no_grad():
                response_token_ids = model.generate(input_ids=input_ids, generation_config=generation_config)
                generated_texts = tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)
            gen_texts.extend(generated_texts)

            # Evaluate humanity scores for the generated texts
            humanity_scores_batch = probability_of_human_generated(generated_texts)
            humanity_scores.extend(humanity_scores_batch)

            # Delete MPS/GPU Tensors
            del input_ids

        except Exception as e:
            print(f"Error generating responses for batch starting at index {i}: {e}")

    # Compute mean & std on the CPU for compatibility
    if humanity_scores:
        mean = np.mean(humanity_scores)
        std = np.std(humanity_scores)
    else:
        mean, std = None, None 
    
    # Clean up by deleting the dataset if needed
    del dataset

    return mean, std, gen_texts


def ppo_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])