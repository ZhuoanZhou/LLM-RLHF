import torch
import csv
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from functools import partial

from huggingface_hub import login
login(token="your_hf_token_here")

max_new_tokens = 512

#output_dir = "results/llama2_sft_7b_chat/final_checkpoint" 
#tokenizer_dir = "meta-llama/Llama-2-7b-chat-hf"

output_dir = "models/tuning_llama_rlstep_850"
tokenizer_dir = "models/tuning_llama_rlstep_850"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(output_dir, device_map = 'auto')

data_files = {"train": "ppo_data/test.csv"}
dataset_original = load_dataset("Misaka19487/pennmutual", data_files=data_files, use_auth_token=True, split='train')
dataset_original = dataset_original

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def create_prompt_formats(sample):
    sample["text"] = "Q: " + sample['prompt'] + " Provide the name of the range at the end.\nA:"
    return sample

print("Preprocessing dataset...")
dataset = dataset_original.map(create_prompt_formats)#, batched=True)

# Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
_preprocessing_function = partial(preprocess_batch, max_length=1024, tokenizer=tokenizer)
dataset = dataset.map(
    _preprocessing_function,
    batched=True,
    #remove_columns=["sample_id", "Ranges", "number", "prompt", "response", "ground_truth", "CoT_demo_prompt", "text"],
)

# Filter out samples that have input_ids exceeding max_length
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < 1024)
dataset = dataset['text']

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch processing function
def process_batch(batch, model, tokenizer, max_length=1024):
    inputs = tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1, num_return_sequences=1)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Define batch size
batch_size = 8  # Adjust this based on your GPU/CPU memory

# Process the entire dataset in batches
results = []
for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    batch_results = process_batch(batch, model, tokenizer)
    results.extend(batch_results)

df = pd.DataFrame(dataset_original)
df['ppo_output'] = results
df.to_csv("data/test_ppo_llama2_7b_chat.csv",index=False)