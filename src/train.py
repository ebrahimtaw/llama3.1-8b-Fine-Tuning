# fine-tuning the Llama-3.1-8B-Instruct on amazon_polarity
# using Unsloth + QLoRA + TRL SFTTrainer

import os
import torch
from dataclasses import dataclass
from dotenv import load_dotenv

from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel

# =========================================================
# Loading the environment variables
# =========================================================
load_dotenv("private.env")
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN not found in private.env"

# =========================================================
# config
# =========================================================
@dataclass
class CFG:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_path: str = "data/amazon_polarity_sft"
    output_dir: str = "outputs/llama31-amazonpol-qlora"

    max_seq_length: int = 304
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0

    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    logging_steps: int = 25
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    bf16: bool = True
    seed: int = 42

CFG = CFG()

# =========================================================
# loading the dataset (prepared earlier)
# =========================================================
print("Loading dataset from disk...")
dataset = load_from_disk(CFG.dataset_path)
print(dataset)

# Must contain "text" column
assert "text" in dataset["train"].column_names

# =========================================================
# loading the model with Unsloth (QLoRA)
# =========================================================
print("Loading model with Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CFG.base_model,
    max_seq_length = CFG.max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = True,
    token = HF_TOKEN,
)

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# applying QLoRA
# =========================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = CFG.seed,
)

# =========================================================
# training arguments
# =========================================================
training_args = TrainingArguments(
    output_dir = CFG.output_dir,
    per_device_train_batch_size = CFG.per_device_train_batch_size,
    gradient_accumulation_steps = CFG.gradient_accumulation_steps,
    num_train_epochs = CFG.num_train_epochs,

    learning_rate = CFG.learning_rate,
    warmup_ratio = CFG.warmup_ratio,
    weight_decay = CFG.weight_decay,

    logging_steps = CFG.logging_steps,
    evaluation_strategy = "steps",
    eval_steps = CFG.eval_steps,
    save_steps = CFG.save_steps,
    save_total_limit = CFG.save_total_limit,

    bf16 = CFG.bf16,
    optim = "adamw_8bit",
    lr_scheduler_type = "cosine",

    report_to = "none",
    seed = CFG.seed,
)

# =========================================================
# trainer (assistant-only loss)
# =========================================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = CFG.max_seq_length,
    packing = True,  # packs multiple samples → faster on A100
    args = training_args,
)

# =========================================================
# train
# =========================================================
print("Starting training...")
trainer.train()

# =========================================================
# saving the final adapter
# =========================================================
print("Saving model...")
trainer.save_model(CFG.output_dir)
tokenizer.save_pretrained(CFG.output_dir)

print("Training complete.")
