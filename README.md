# Llama 3.1 8B QLoRA Fine-Tuning — Amazon Polarity Dataset

This project fine-tunes the **Meta-Llama-3.1-8B-Instruct with (4-bit) quantization** using **Unsloth** and **QLoRA** for sentiment classification on the **Amazon Polarity** dataset.

## Structure
- `src/` — contains modular training, evaluation, and inference scripts
- `notebooks/` — the full Jupyter Notebook used for the fine-tuning
- `requirements.txt` — dependencies required for the data prep + training
- `.env` — this should contain my HF_TOKEN (so I excluded it from Git), create one and store yours

## Run
```bash
python -m src.train
