import os
import random
import numpy as np
from dataclasses import dataclass
from collections import Counter

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

@dataclass
class CFG:
    dataset_id: str = "fancyzhx/amazon_polarity"
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    out_dir: str    = "checkpoints/qlora-llama31-8b-amazonpol"

    max_len: int   = 304      # divisible by 8 (Tensor Core friendly)
    pad_mult: int  = 8

    batch_size: int = 1
    grad_accum: int = 16
    lr: float       = 2e-4
    epochs: float   = 1.0

    logging_steps: int = 50
    eval_steps: int    = 1000
    save_steps: int    = 1000

CFG

ds = load_dataset(CFG.dataset_id)

def label_stats(split, name):
    c = Counter(int(r["label"]) for r in split)
    total = sum(c.values())
    print(f"{name} size = {total:,}")
    for k in sorted(c):
        print(f"  label {k}: {c[k]:,} ({100*c[k]/total:.1f}%)")

label_stats(ds["train"], "FULL train")
label_stats(ds["test"],  "FULL test")

N_TRAIN_POOL = 270_000   # 250k train + 20k val
N_TEST_DEV  = 40_000    # quick dev-time test

def stratified_sample(hfds_split, n_total, label_col="label", seed=SEED):
    hfds_split = hfds_split.shuffle(seed=seed)
    idx_by_label = {}

    for i, ex in enumerate(hfds_split):
        idx_by_label.setdefault(int(ex[label_col]), []).append(i)

    labels = sorted(idx_by_label.keys())
    per_label = {lab: int(round(n_total / len(labels))) for lab in labels}

    while sum(per_label.values()) > n_total:
        per_label[labels[-1]] -= 1
    while sum(per_label.values()) < n_total:
        per_label[labels[0]] += 1

    chosen = []
    for lab in labels:
        inds = idx_by_label[lab]
        random.Random(seed + lab).shuffle(inds)
        chosen.extend(inds[:per_label[lab]])

    random.Random(seed).shuffle(chosen)
    return hfds_split.select(chosen)

train_pool = stratified_sample(ds["train"], N_TRAIN_POOL)
test_dev   = stratified_sample(ds["test"],  N_TEST_DEV)

label_stats(train_pool, "TRAIN POOL")
label_stats(test_dev,   "TEST DEV")

print("Example record:", {k: train_pool[0][k] for k in ["label", "title", "content"]})

@dataclass
class CHAT:
    system: str = (
        "You are a sentiment classification assistant. "
        "Reply with exactly one word: Positive or Negative. "
        "Do not add punctuation or explanations."
    )

LABEL_TEXT = {0: "Negative", 1: "Positive"}
MAX_CHARS = 1500  # trim long reviews early

def to_chat_messages(ex):
    title   = (ex.get("title")   or "").strip()
    content = (ex.get("content") or "").strip()
    review  = (title + ("\n\n" if title and content else "") + content).strip()
    review  = review[:MAX_CHARS]

    return {
        "messages": [
            {"role": "system",    "content": CHAT.system},
            {"role": "user",      "content": f"Review:\n{review}\n\nLabel (Positive or Negative):"},
            {"role": "assistant", "content": LABEL_TEXT[int(ex["label"])]},
        ]
    }

train_pool_chat = train_pool.map(
    to_chat_messages,
    remove_columns=train_pool.column_names
)

test_dev_chat = test_dev.map(
    to_chat_messages,
    remove_columns=test_dev.column_names
)

print("Chat example:\n", train_pool_chat[0]["messages"])

load_dotenv("private.env")
hf_token = os.getenv("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(
    CFG.base_model,
    use_fast=True,
    token=hf_token,
)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Pad token:", tok.pad_token, "| id:", tok.pad_token_id)
print("Chat template preview:\n", (tok.chat_template or "")[:400])

def render_chat(batch):
    texts = []
    for msgs in batch["messages"]:
        txt = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(txt)
    return {"text": texts}

train_pool_txt = train_pool_chat.map(
    render_chat,
    batched=True,
    remove_columns=train_pool_chat.column_names
)

test_dev_txt = test_dev_chat.map(
    render_chat,
    batched=True,
    remove_columns=test_dev_chat.column_names
)

print("Rendered sample:\n", train_pool_txt[0]["text"][:400])

SAMPLE_N = 8
enc = tok(
    [train_pool_txt[i]["text"] for i in range(SAMPLE_N)],
    truncation=True,
    max_length=CFG.max_len,
    padding=False,
)

lengths = [len(x) for x in enc["input_ids"]]
print("Token lengths:", lengths)
print("min / mean / max:", min(lengths), f"{np.mean(lengths):.1f}", max(lengths))

print("\nDecoded tail:\n", tok.decode(enc["input_ids"][0][-20:]))

train_pool_txt = train_pool_txt.shuffle(seed=SEED)
test_dev_txt   = test_dev_txt.shuffle(seed=SEED)

N_TRAIN, N_VAL = 250_000, 20_000
assert len(train_pool_txt) >= N_TRAIN + N_VAL

train_text = train_pool_txt.select(range(N_TRAIN))
val_text   = train_pool_txt.select(range(N_TRAIN, N_TRAIN + N_VAL))
test_text  = test_dev_txt

small_text = DatasetDict(
    train=train_text,
    validation=val_text,
    test=test_text,
)

print(small_text)
print("Sizes → train / val / test:", len(train_text), len(val_text), len(test_text))

SYSTEM_MSG = CHAT.system
dummy_user = (
    "Review:\nThis is a dummy review used to extract the assistant header.\n\n"
    "Label (Positive or Negative):"
)

msgs_user = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user",   "content": dummy_user},
]

user_prompt = tok.apply_chat_template(
    msgs_user,
    tokenize=False,
    add_generation_prompt=True,
)

user_plus_assistant_header = tok.apply_chat_template(
    msgs_user + [{"role": "assistant", "content": ""}],
    tokenize=False,
    add_generation_prompt=False,
)

response_template = user_plus_assistant_header[len(user_prompt):]
print("Response template preview:", response_template.replace("\n", "\\n")[:160])

sample_txt = small_text["train"][0]["text"]
print("Header found in sample →", response_template in sample_txt)

def find_sublist(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

enc_full = tok(sample_txt, add_special_tokens=False)
enc_resp = tok(response_template, add_special_tokens=False)

pos = find_sublist(enc_full["input_ids"], enc_resp["input_ids"])
print("Assistant header token index:", pos, "(>=0 is correct)")

# =========================================================
# this is for saving the dataset to disk for training (RunPod-compatible)
# =========================================================
SAVE_PATH = "data/amazon_polarity_sft"

os.makedirs(SAVE_PATH, exist_ok=True)
small_text.save_to_disk(SAVE_PATH)

print(f"Dataset saved to: {SAVE_PATH}")
