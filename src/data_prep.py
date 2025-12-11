!pip install -q -U \
  "unsloth>=2025.10.1" \
  "transformers>=4.56.0,<=4.56.2" \
  "datasets>=3.4.0,<3.5.0" \
  "accelerate>=0.34.1" \
  "peft>=0.13.0" \
  "bitsandbytes>=0.45.0" \
  "trl>=0.12.0" \
  "sentencepiece>=0.2.0" \
  "evaluate>=0.4.2" \
  "numpy>=1.26.4,<2.0.0" \
  "pandas==2.2.2" \
  "scikit-learn>=1.5.0,<1.7.0" \
  "matplotlib>=3.9.0,<3.10.0" \
  "pyarrow>=16.1.0,<20.0.0" \
  "protobuf>=5.29.1,<6.0.0" \
  "fsspec>=2024.3.0,<2024.12.0" \
  "python-dotenv" \
  "huggingface_hub[hf_xet]" \
  --extra-index-url https://jllllll.github.io/bitsandbytes-wheels/cu121

import random, numpy as np
from dataclasses import dataclass
from datasets import load_dataset, Dataset, DatasetDict
import os
from dotenv import load_dotenv
from collections import Counter
import random
from dataclasses import dataclass
from transformers import AutoTokenizer
import os
import matplotlib.pyplot as plt
from datasets import DatasetDict

SEED = 42
random.seed(SEED); np.random.seed(SEED)

@dataclass
class CFG:
    dataset_id: str = "amazon_polarity"
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # swap to Mistral/Qwen if lacking access
    out_dir: str   = "checkpoints/qlora-llama8b-amazonpol"
    max_len: int   = 300   # reviews: 300 is enough to fit the 8GB gpu
    pad_mult: int  = 8
    batch_size: int = 1
    grad_accum: int = 16   # effective batch ~= 16
    lr: float      = 2e-4
    epochs: float  = 1.0
    logging_steps: int = 50
    eval_steps: int    = 1000
    save_steps: int    = 1000

CFG

ds = load_dataset("amazon_polarity")

load_dotenv("private.env")
hf_token = os.getenv("HF_TOKEN")

def label_stats(split, name):
    c = Counter(int(r["label"]) for r in split)
    total = sum(c.values())
    print(f"{name} size = {total:,}")
    for k in sorted(c):
        pct = 100 * c[k] / total
        print(f"  label {k}: {c[k]:,}  ({pct:.1f}%)")

label_stats(ds["train"], "FULL train")
label_stats(ds["test"],  "FULL test")

SEED = 42
random.seed(SEED)

N_TRAIN_POOL = 270_000  # 250k train + 20k val
N_TEST_DEV   = 40_000   # quick dev-time test

def stratified_sample(hfds_split, n_total, label_col="label", seed=SEED):
    """
    Make a smaller split with the same label distribution (pos/neg).
    Deterministic → same subset every run with the same seed.
    """
    hfds_split = hfds_split.shuffle(seed=seed)
    idx_by_label = {}
    for i, ex in enumerate(hfds_split):
        idx_by_label.setdefault(int(ex[label_col]), []).append(i)

    labels = sorted(idx_by_label.keys())
    per_label = {lab: int(round(n_total / len(labels))) for lab in labels}
    # fix rounding drift
    while sum(per_label.values()) > n_total: per_label[labels[-1]] -= 1
    while sum(per_label.values()) < n_total: per_label[labels[0]]  += 1

    chosen = []
    for lab in labels:
        inds = idx_by_label[lab]
        random.Random(seed + lab).shuffle(inds)
        chosen.extend(inds[:per_label[lab]])

    random.Random(seed).shuffle(chosen)
    return hfds_split.select(chosen)

train_pool = stratified_sample(ds["train"], N_TRAIN_POOL)
test_dev   = stratified_sample(ds["test"],  N_TEST_DEV)

print(train_pool)
print(test_dev)
# sanity: check balance of the reduced sets
label_stats(train_pool, "TRAIN POOL (270k)")
label_stats(test_dev,   "TEST DEV (40k)")
print("Example:", {k: train_pool[0][k] for k in ["label","title","content"]})

@dataclass
class CHAT:
    system: str = "You are a helpful sentiment assistant. Reply with exactly one word: Positive or Negative."

LABEL_TEXT = {0: "Negative", 1: "Positive"}

def to_chat_messages(ex):
    title   = (ex.get("title")   or "").strip()
    content = (ex.get("content") or "").strip()
    review  = (title + ("\n\n" if title and content else "") + content).strip()

    return {
        "messages": [
            {"role":"system",   "content": CHAT.system},
            {"role":"user",     "content": f"Review:\n{review}\n\nLabel (Positive or Negative):"},
            {"role":"assistant","content": LABEL_TEXT[int(ex["label"])]},
        ]
    }

train_pool_chat = train_pool.map(to_chat_messages, remove_columns=train_pool.column_names)
test_dev_chat   = test_dev.map(to_chat_messages,   remove_columns=test_dev.column_names)

print("One chat example:\n", train_pool_chat[0]["messages"])

tok = AutoTokenizer.from_pretrained(
    CFG.base_model,
    use_fast=True,
    trust_remote_code=True,
    token=os.getenv("HF_TOKEN")
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Pad token:", tok.pad_token, " | id:", tok.pad_token_id)
print("\nChat template head:\n", (tok.chat_template or "")[:400])

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

train_pool_txt = train_pool_chat.map(render_chat, batched=True, remove_columns=train_pool_chat.column_names)
test_dev_txt   = test_dev_chat.map(render_chat,   batched=True, remove_columns=test_dev_chat.column_names)

print("Rendered text sample (truncate):\n", train_pool_txt[0]["text"][:400])
print("Lengths (chars) — first 3:", [len(train_pool_txt[i]["text"]) for i in range(3)])

SAMPLE_N = 8  # preview a handful of samples
enc = tok([train_pool_txt[i]["text"] for i in range(SAMPLE_N)],
          truncation=True, max_length=CFG.max_len, padding=False)

lengths = [len(x) for x in enc["input_ids"]]
print("Token lengths (first", SAMPLE_N, "):", lengths)
print("min/mean/max:", min(lengths), f"{sum(lengths)/len(lengths):.1f}", max(lengths))

# Peek at the tail of sample 0 to confirm the assistant label is included
print("\nDecoded tail of sample 0:\n", tok.decode(enc["input_ids"][0][-20:]))

SEED = 42
train_pool_txt = train_pool_txt.shuffle(seed=SEED)
test_dev_txt   = test_dev_txt.shuffle(seed=SEED)

N_TRAIN, N_VAL = 250_000, 20_000
assert len(train_pool_txt) >= N_TRAIN + N_VAL

train_text = train_pool_txt.select(range(N_TRAIN))
val_text   = train_pool_txt.select(range(N_TRAIN, N_TRAIN + N_VAL))
test_text  = test_dev_txt  # the 40k we built

small_text = DatasetDict(train=train_text, validation=val_text, test=test_text)
print(small_text)
print("Sizes → train/val/test(dev):", len(train_text), len(val_text), len(test_text))
print("Full test kept for final report:", len(ds["test"]))\

# this is for building a canonical dummy example using the real system message,
# and a dummy user text (the exact user content doesn’t matter here).
SYSTEM_MSG = "You are a helpful sentiment assistant. Reply with exactly one word: Positive or Negative."
dummy_user = "Review:\nThis is a dummy review used to extract the assistant header.\n\nLabel (Positive or Negative):"

msgs_user = [
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user",   "content": dummy_user},
]

# 1) Render up to the assistant header (no assistant message yet)
user_prompt = tok.apply_chat_template(
    msgs_user,
    tokenize=False,
    add_generation_prompt=True,   # <- append the assistant header, ready to generate
)

# 2) render with an *empty* assistant message (same situation as training strings before the label)
user_plus_assistant_header = tok.apply_chat_template(
    msgs_user + [{"role": "assistant", "content": ""}],
    tokenize=False,
    add_generation_prompt=False,
)

# 3) the assistant header we need is the difference between the two renders
response_template = user_plus_assistant_header[len(user_prompt):]

print("Response template (escaped preview):", response_template.replace("\n","\\n")[:160], "...")

# this is for verifying it exists in the real rendered training text
sample_txt = small_text["train"][0]["text"]
print("Does response_template occur in sample string? ->", response_template in sample_txt)

# Token-level verification (most important for TRL)
def find_sublist(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

enc_full = tok(sample_txt, add_special_tokens=False)
enc_resp = tok(response_template, add_special_tokens=False)
pos = find_sublist(enc_full["input_ids"], enc_resp["input_ids"])
print("Prefix token sequence found at index:", pos, "(>=0 is good)")
