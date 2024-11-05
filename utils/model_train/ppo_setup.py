import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from utils.model_train.metrics import get_all_metrics
from utils.model_train.model_train_utils import (
    build_dataset, trainable_model_parameters, evaluate_humanity
)
from utils.common.helper_funcs import setup_logging
from utils.common.constants import (
    LEARNING_RATE, MAX_PPO_EPOCHS, MINI_BATCH_SIZE, TRAIN_BATCH_SIZE,
    LORA_RANK_DIMS, SAMPLE
)
import mlflow

def initialize_logging():
    return setup_logging()

def load_tokenizer_and_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def setup_ppo_model(model, device, lora_rank_dims):
    # Configure LoRA and prepare PPO model
    from peft import get_peft_model, LoraConfig, TaskType

    lora_config = LoraConfig(
        r=lora_rank_dims,
        lora_alpha=lora_rank_dims,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    peft_model = get_peft_model(model, lora_config).to(device)
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        peft_model, torch_dtype=torch.bfloat16, is_trainable=True
    )

    ref_model = create_reference_model(ppo_model).to(device)
    return ppo_model, ref_model

def prepare_dataset(logger, tokenizer):
    return build_dataset(logger, tokenizer=tokenizer, sample=SAMPLE)

def setup_mlflow(experiment_name):
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "does not exist" in str(e):
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        else:
            raise