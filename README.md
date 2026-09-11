# Llama 3.1 8B QLoRA Fine-Tuning — Amazon Polarity Dataset

This project fine-tunes the **Meta-Llama-3.1-8B-Instruct with (4-bit) quantization** using **Unsloth** and **QLoRA** for sentiment classification on the **Amazon Polarity** dataset.

I wrote the logic in a jupyter notebook then I separated it into a data prep and train scripts. You can use the scripts to fine-tune the model, I've used RunPod to rent an H100 GPU to do the fine-tuning. The next step is to get it served via vLLM on the H100 (maybe).
