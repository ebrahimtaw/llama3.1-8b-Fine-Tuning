from unsloth import FastLanguageModel

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 300, # i changed it from 256 to 300
    dtype = None,             # auto-detect (uses bfloat16 if available)
    load_in_4bit = True,      # 4-bit quantization
)

# quick check
print("Model loaded:", model_name)
print("Pad token:", tokenizer.pad_token)
print("Device:", model.device)

from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                  # rank
    lora_alpha = 64,         # scaling
    lora_dropout = 0.05,     # dropout
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # saves VRAM
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

model.print_trainable_parameters()

# small_text should already be my DatasetDict with 'text' column: train/validation/test
print(small_text)

# peek a couple of rendered rows (already chat-templated strings)
for i in range(2):
    print("--- sample", i, "---")
    print(small_text["train"][i]["text"][:400], "...\n")

import torch
from trl import SFTTrainer, SFTConfig

# T4 prefers fp16; bf16 usually not available
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

train_args = SFTConfig(
    output_dir                    = "/content/qlora-llama31-8b-unsloth-amazonpol",
    num_train_epochs              = 1,          # starting with 1 epoch; I can do 2 later
    per_device_train_batch_size   = 1,
    per_device_eval_batch_size    = 1,
    gradient_accumulation_steps   = 32,         # I chose this which simulates an effective batch of 32
    learning_rate                 = 2e-4,
    lr_scheduler_type             = "cosine",
    warmup_ratio                  = 0.03,
    logging_steps                 = 50,
    save_steps                    = 1000,       # frequent checkpoints
    eval_strategy                 = "steps",
    eval_steps                    = 1000,
    max_seq_length                = 256,        # this keeps VRAM low
    packing                       = False,      # this one MUST be False for completion-only loss
    fp16                          = True,
    bf16                          = False,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = small_text["train"],
    eval_dataset       = small_text["validation"],
    dataset_text_field = "text",
    response_template  = response_template,   # this is the verified assistant header snippet
    args               = train_args,
)

print("Trainer ready. Starting training…")
train_output = trainer.train()
train_output
