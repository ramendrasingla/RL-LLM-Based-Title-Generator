import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from tqdm import tqdm
from accelerate import Accelerator
from trl import PPOTrainer, PPOConfig
import mlflow

from utils.model_train.ppo_setup import initialize_logging, load_tokenizer_and_model, setup_ppo_model, prepare_dataset, setup_mlflow
from utils.model_train.metrics import get_all_metrics
from utils.model_train.model_train_utils import evaluate_humanity, probability_of_human_generated, ppo_collator
from utils.common.constants import LEARNING_RATE, MAX_PPO_EPOCHS, MINI_BATCH_SIZE, TRAIN_BATCH_SIZE, LORA_RANK_DIMS, generation_kwargs

# Set up MLflow and logging
experiment_name = 'RL_LLM_Title_Generator'
setup_mlflow(experiment_name)
logger = initialize_logging()

# Setup device and load model/tokenizer
device = torch.device('mps' if torch.mps.is_available() else ('gpu' if torch.cuda.is_available() else 'cpu'))
model_name = "google/flan-t5-large"
tokenizer, model = load_tokenizer_and_model(model_name, device)

# Prepare PPO and reference models
ppo_model, ref_model = setup_ppo_model(model, device, LORA_RANK_DIMS)

# Load dataset
dataset = prepare_dataset(logger, tokenizer)

# Initialize Accelerator, PPOConfig, and PPOTrainer
accelerator = Accelerator()
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

# Training loop with MLflow logging
with mlflow.start_run(run_name="RL_PPO_training"):
    for epoch in range(config.ppo_epochs):
        print(f"\n=== Starting Epoch {epoch + 1}/{config.ppo_epochs} ===\n")
        
        for step, batch in enumerate(ppo_trainer.dataloader):
            prompt_tensors = [t.to(torch.long).to(device) for t in batch["input_ids"]]
            title_tensors = [t.to(torch.long).to(device) for t in ppo_trainer.generate(prompt_tensors, **generation_kwargs)]

            batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in title_tensors]
            batch_humanity_scores = probability_of_human_generated(batch["response"])
            reward_tensors = [torch.tensor(t, dtype=torch.float32).to(device) for t in batch_humanity_scores]

            stats = accelerator.unwrap_model(ppo_trainer).step(prompt_tensors, title_tensors, reward_tensors)
            
            if (step + 1) % 10 == 0 or step == len(ppo_trainer.dataloader) - 1:
                accelerator.print(f"Logging stats for step {step + 1}...")
                ppo_trainer.log_stats(stats, batch, reward_tensors)

            del prompt_tensors, title_tensors, reward_tensors

        # Evaluation after each epoch
        mean, std, predicted_title = evaluate_humanity(
            model=accelerator.unwrap_model(ppo_trainer.model), tokenizer=tokenizer, dataset=dataset["test"]
        )
        linguistic_metrics = get_all_metrics(predicted_title, dataset["test"]["title"])

        # Log metrics to MLflow
        mlflow.log_metric("mean_humanity_score", mean, step=epoch + 1)
        mlflow.log_metric("std_humanity_score", std, step=epoch + 1)
        for k, v in linguistic_metrics.items():
            mlflow.log_metric(k, v, step=epoch + 1)

        # Save model checkpoint
        mlflow.pytorch.log_model(accelerator.unwrap_model(ppo_trainer.model), f"model_epoch_{epoch + 1}")
        print(f"\nEnd of epoch {epoch + 1} - mean humanity score: {mean:.4f}, std: {std:.4f}\n")
