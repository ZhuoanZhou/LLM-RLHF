import torch
import csv
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from functools import partial

#model_name = "meta-llama/Llama-2-13b-chat-hf" 
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')

output_dir = "results/option_2_added_prompt/final_merged_checkpoint"
tokenizer_dir = "results/option_2_added_prompt/final_merged_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(output_dir, device_map = 'auto')

#prompt = """What are the soft skills in the following sentence if there are any

#Sentence:
#Member of a team that has shipped production code at scale.

#Soft skills:"""

data_files = {"train": "test.csv"}
dataset_original = load_dataset("Misaka19487/pennmutual", data_files=data_files, use_auth_token=True, split='train')

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
    sample["text"] = sample['CoT_demo_prompt']
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

#print(dataset)

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Batch processing function
def process_batch(batch, model, tokenizer, max_length=1024):
    inputs = tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.1, num_return_sequences=1)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Define batch size
batch_size = 8  # Adjust this based on your GPU/CPU memory

# Process the entire dataset in batches
results = []
for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    batch_results = process_batch(batch, model, tokenizer)
    results.extend(batch_results)

#print(results)

df = pd.DataFrame(dataset_original)
df['option_2_output'] = results
df.to_csv("data/test_finetuned_option_2_output.csv",index=False)


#print(outputs)
#print(len(outputs))
#print(outputs[0])


"""
test_df = pd.read_csv("data/test.csv")

llama_outputs_1 = []
llama_outputs_2 = []

for i, row in tqdm(test_df.iterrows(), total = len(test_df)):
    prompt = row['prompt']
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    print(input_ids)
    generated_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True, temperature=0.1, num_return_sequences=1)
    out = tokenizer.batch_decode(generated_ids)
    llama_outputs_1.append(out)

    print(out)

    prompt = row['CoT_demo_prompt']
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True, temperature=0.1, num_return_sequences=1)
    out = tokenizer.batch_decode(generated_ids)
    llama_outputs_2.append(out)

    print(out)

    break


test_df['option_1_output'] = llama_outputs_1
test_df['option_2_output'] = llama_outputs_2

test_df.to_csv("direct_infer_result.csv", index = False)
"""

"""
start_time = time.time()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
generated_ids = model.generate(input_ids, max_new_tokens=50,  do_sample=True, temperature=0.5, num_return_sequences=4, num_beams = 4)
#generated_ids = model.generate(input_ids, max_new_tokens=50,  do_sample=True, temperature=0.5, num_return_sequences=4, num_beams = 4, eos_token_id=[2, 13])
#generated_ids = model.generate(input_ids, max_new_tokens=50,  do_sample=True, temperature=0.5, num_return_sequences=4, num_beams = 4, eos_token_id=[1, 835, 2796])

out = tokenizer.batch_decode(generated_ids)
print(out)

print("--- %s seconds ---" % (time.time() - start_time))
"""
