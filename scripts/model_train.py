import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
from peft import get_peft_model, LoraConfig, PeftModel, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm
tqdm.pandas()

from utils.metrics import (reward_function, count_adjectives, word_diversity,
                     get_sentiment, pos_pattern_matching)
from utils.train_utils import (build_dataset, CastOutputToFloat, load_reward_model,
                         trainable_model_parameters, evaluate_humanity, 
                         ppo_collator)
from utils.constants import LORA_RANK_DIMS

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the T5-Small model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.generation_config.pad_token_id = tokenizer.pad_token_id

# Get the dataset
dataset = build_dataset(tokenizer=tokenizer, sample=100)

# Configuration for LoRA
lora_config = LoraConfig(
    r=LORA_RANK_DIMS,  # Rank
    lora_alpha=LORA_RANK_DIMS,
    target_modules=["attn.c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

max_new_tokens=2000

input_text = dataset['train']['query'][0]

input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        
generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                        top_k=0.0,
                                        top_p=1.0,
                                        do_sample=True)

response_token_ids = model.generate(input_ids=input_ids,
                                    generation_config=generation_config)

generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)

print(generated_text)
print("*"*50)
print(dataset['train']['title'][0])

# # Load or initialize the PEFT model with LoRA configuration
# peft_model = get_peft_model(model, lora_config).to(device=device)

# # Print number of trainable parameters for LoRA adaptation
# print(f'PEFT model parameters to be updated:\n{trainable_model_parameters(peft_model)}\n')

# ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,                                                               
#                                                                torch_dtype=torch.bfloat16,
#                                                                is_trainable=True)

# print(f'PPO model parameters to be updated (ValueHead + 769 params):\n{trainable_model_parameters(ppo_model)}\n')
# print(ppo_model.v_head)

# ref_model = create_reference_model(ppo_model)

# print(f'Reference model parameters to be updated:\n{trainable_model_parameters(ref_model)}\n')

# mean_before_fine_tuning, std_before_fine_tuning = evaluate_humanity(model=ref_model, 
#                                                                           tokenizer=tokenizer, 
#                                                                           dataset=dataset["test"], 
#                                                                           num_samples=10)

# print(f'Humanity [mean, std] before fine-tuning: [{mean_before_fine_tuning}, {std_before_fine_tuning}]')

# learning_rate=1.41e-5
# max_ppo_epochs=1
# mini_batch_size=4
# batch_size=16

# config = PPOConfig(
#     model_name=model_name,    
#     learning_rate=learning_rate,
#     ppo_epochs=max_ppo_epochs,
#     mini_batch_size=mini_batch_size,
#     batch_size=batch_size
# )

# ppo_trainer = PPOTrainer(config=config, 
#                          model=ppo_model, 
#                          ref_model=ref_model, 
#                          tokenizer=tokenizer, 
#                          dataset=dataset["train"], 
#                          data_collator=ppo_collator)

# output_min_length = 100
# output_max_length = 400
# max_ppo_steps = 10
# output_length_sampler = LengthSampler(output_min_length, output_max_length)

# generation_kwargs = {
#     "min_length": 5,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True
# }

# reward_pipe, reward_kwargs = load_reward_model()

# for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
#     # Break when you reach max_steps.
#     if step >= max_ppo_steps:
#         break   

#     prompt_tensors = batch["input_ids"]

#     # Get response from FLAN-T5/PEFT LLM.
#     summary_tensors = []

#     for prompt_tensor in prompt_tensors:
#         max_new_tokens = output_length_sampler()        
            
#         generation_kwargs["max_new_tokens"] = max_new_tokens
#         summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
        
#         summary_tensors.append(summary.squeeze()[-max_new_tokens:])
        
#     # This needs to be called "response".
#     batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

#     # Compute reward outputs. 
#     rewards = reward_pipe(batch["response"], **reward_kwargs)

#     # You use the `human_generated_index` item because this is the score for the positive `human_generated_index` class.
#     human_generated_index = 1
#     reward_tensors = [torch.tensor(reward[human_generated_index]["score"]) for reward in rewards]    

#     # Run PPO step.
#     stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
#     ppo_trainer.log_stats(stats, batch, reward_tensors)
    
#     print(f'objective/kl: {stats["objective/kl"]}')
#     print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
#     print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
#     print('-'.join('' for x in range(100)))