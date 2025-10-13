# Llama 3.1 8B QLoRA Fine-Tuning — Amazon Polarity Dataset

This project fine-tunes **Meta-Llama-3.1-8B-Instruct (4-bit)** using **Unsloth** and **QLoRA** for sentiment classification on the **Amazon Polarity** dataset.

## Structure
- `src/` — modular training, evaluation, and inference scripts
- `notebooks/` — exploratory Jupyter experiments
- `requirements.txt` — dependencies for Colab / local setup
- `.env` — this should contain my HF_TOKEN (so I excluded it from Git)

## Run
```bash
python -m src.train
