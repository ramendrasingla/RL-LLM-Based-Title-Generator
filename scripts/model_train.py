import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
# Mlflow logging and registry
import mlflow
# tqdm library makes the loops show a smart progress meter.
tqdm.pandas()

from utils.model_training.metrics import get_all_metrics
from utils.model_training.model_train_utils import (build_dataset, probability_of_human_generated,
                                     trainable_model_parameters, evaluate_humanity, 
                                     ppo_collator, load_latest_model)
from utils.common.helper_funcs import setup_logging
from utils.common.constants import (   LEARNING_RATE,
                                MAX_PPO_EPOCHS,
                                MINI_BATCH_SIZE,
                                TRAIN_BATCH_SIZE,
                                MAX_NEW_TOKENS,
                                LORA_RANK_DIMS,
                                generation_kwargs,
                                SAMPLE)

# Set up MLflow
experiment_name = 'RL_LLM_Title_Generator'

# Check if the experiment already exists
try:
    mlflow.set_experiment(experiment_name)
except mlflow.exceptions.MlflowException as e:
    if "does not exist" in str(e):
        # Create a new experiment if it does not exist
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    else:
        raise

# Setup Logging
global logger
logger = setup_logging()

# Setup device
device = torch.device('mps' if torch.mps.is_available() else ('gpu' if torch.gpu.is_available() else 'cpu'))

# Load model
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.generation_config.pad_token_id = tokenizer.pad_token_id


# Get the dataset
dataset = build_dataset(logger, tokenizer=tokenizer, sample=SAMPLE)

# Configuration for LoRA
lora_config = LoraConfig(
    r=LORA_RANK_DIMS,  # Rank
    lora_alpha=LORA_RANK_DIMS,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Load or initialize the PEFT model with LoRA configuration
peft_model = get_peft_model(model, lora_config).to(device=device)

ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=True)

print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{trainable_model_parameters(ppo_model)}\n')
print(ppo_model.v_head)

ref_model = create_reference_model(ppo_model).to(device)

# Evaluate the model at the end of the epoch
mean_before_fine_tuning, std_before_fine_tuning, ref_model_predicted_title = evaluate_humanity(model = ref_model, 
                                                                                               tokenizer = tokenizer, 
                                                                                               dataset = dataset["test"])
linguistic_metrics_before_fine_tuning = get_all_metrics(ref_model_predicted_title, dataset["test"]["title"])

print(f'Humanity [mean, std] before fine-tuning: [{mean_before_fine_tuning}, {std_before_fine_tuning}]')

# Start MLflow logging
with mlflow.start_run(run_name="ref_model"):

    # Log the reference model humanity scores
    mlflow.log_metric("mean_humanity_score", mean_before_fine_tuning, step=0)
    mlflow.log_metric("std_humanity_score", std_before_fine_tuning, step=0)

    for k, v in linguistic_metrics_before_fine_tuning.items():
        mlflow.log_metric(k, v, step=0)

    # Log the reference model
    mlflow.pytorch.log_model(ref_model, "ref_model")

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Initialize PPO config and trainer
config = PPOConfig(
    model_name=model_name,    
    learning_rate=LEARNING_RATE,
    ppo_epochs=MAX_PPO_EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    batch_size=TRAIN_BATCH_SIZE
)

ppo_trainer = PPOTrainer(
    config=config, 
    model=ppo_model, 
    ref_model=ref_model, 
    tokenizer=tokenizer, 
    dataset=dataset["train"], 
    data_collator=ppo_collator
)

# Wrap dataloader and model for multi-GPU
ppo_trainer.dataloader = accelerator.prepare(ppo_trainer.dataloader)
ppo_trainer.model = accelerator.prepare(ppo_trainer.model)

# Optional: Load the latest model from MLflow
# load_latest_model(ppo_trainer)

with mlflow.start_run():
    print("Starting PPO training and logging with MLflow...")

    for epoch in range(ppo_trainer.config.ppo_epochs):
        print(f"\n=== Starting Epoch {epoch + 1}/{ppo_trainer.config.ppo_epochs} ===\n")
        
        for step, batch in enumerate(ppo_trainer.dataloader):
            # Move inputs to the relevant device using accelerator
            prompt_tensors = [t.to(torch.long).to(device) for t in batch["input_ids"]]

            print(f"Generating responses for step {step + 1}...")
            # Generate and move responses to device
            title_tensors = [t.to(torch.long).to(device) for t in ppo_trainer.generate(prompt_tensors, **generation_kwargs)]

            batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in title_tensors]

            print("Calculating rewards in batch...")
            batch_humanity_scores = probability_of_human_generated(batch["response"])
            reward_tensors = [torch.tensor(t, dtype=torch.float32).to(device) for t in batch_humanity_scores]

            print("Running PPO step...")
            # Use accelerator with the step function
            stats = accelerator.unwrap_model(ppo_trainer).step(prompt_tensors, title_tensors, reward_tensors)

            if (step + 1) % 10 == 0 or step == len(ppo_trainer.dataloader) - 1:
                accelerator.print(f"Logging stats for step {step + 1}...")
                ppo_trainer.log_stats(stats, batch, reward_tensors)

            # Clear GPU memory
            del prompt_tensors, title_tensors, reward_tensors

        print("Evaluating humanity score for test set...")
        mean, std, predicted_title = evaluate_humanity(
            model=accelerator.unwrap_model(ppo_trainer.model), tokenizer=tokenizer, dataset=dataset["test"]
        )
        
        print("Calculating linguistic metrics...")
        linguistic_metrics = get_all_metrics(predicted_title, dataset["test"]["title"])

        print("Logging metrics to MLflow...")
        mlflow.log_metric("mean_humanity_score", mean, step=epoch + 1)
        mlflow.log_metric("std_humanity_score", std, step=epoch + 1)
        for k, v in linguistic_metrics.items():
            mlflow.log_metric(k, v, step=epoch + 1)

        print(f"Saving model state for epoch {epoch + 1}...")
        mlflow.pytorch.log_model(accelerator.unwrap_model(ppo_trainer.model), f"model_epoch_{epoch + 1}")

        print(f"\nEnd of epoch {epoch + 1} - mean humanity score: {mean:.4f}, std: {std:.4f}\n")
