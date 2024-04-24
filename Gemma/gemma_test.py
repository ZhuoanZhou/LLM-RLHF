# accelerate launch --config_file deepspeed_zero3.yaml --multi_gpu --num_machines 1 --num_processes 4 gemma_test.py

import os
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback
from accelerate import Accelerator
from functools import partial
import bitsandbytes as bnb

# Login to the Hugging Face Hub. You can comment out these lines if your YOUR_HUGGINGFACE_TOKEN is already set.
from huggingface_hub import login
login(token="your_hf_token", add_to_git_credential=True)

model_name = "google/gemma-2b-it"
output_dir = "gemma-2b-it-sft"

# Load my dataset from huggingFace repo
#data_files = {"train": "train_0.csv", "test": "test_0.csv"}
data_files = {"train": "train.csv"}
dataset = load_dataset("Misaka19487/pennmutual", data_files=data_files, use_auth_token=True, split='train')
data_files = {"validation":"validate.csv"}
eval_dataset = load_dataset("Misaka19487/pennmutual", data_files=data_files, use_auth_token=True, split='validation')

print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

print(f'Number of prompts: {len(eval_dataset)}')
print(f'Column names are: {eval_dataset.column_names}')

def create_prompt_formats(sample):

    """
    Format various fields of the sample ('Body', 'tags_3')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """
    """
    temp_t = re.findall(r'\b\d+\b', sample['tags_4'])
    temp_t_list = []
    for t in temp_t:
        temp_t_list.append(soft_skill_list[int(t)])
    response = ", ".join(temp_t_list)

    INTRO_BLURB = "What are the soft skills in the following sentence if there are any?\n"
    INSTRUCTION_KEY = "### Sentence:"
    #INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Soft skills:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['Body']}"
    #input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{response}"
    end = f"{END_KEY}"
    parts = [part for part in [blurb, instruction, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample
    """

    sample["text"] = "Q: " + sample['prompt'] + " Provide the name of the range at the end.\nA: " + sample['response']
    return sample
    


# It reformulate the data into the following format:
'''
What are the soft skills in the following sentence if there are any?

### Sentence:
Looking for an individual who embodies adaptability, can handle various tasks, and maintains precision in work.

### Soft skills:
Adaptability, Multitasking, Attention to detail

### End'''

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        #remove_columns=["context", "response", "text", "category"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

model = AutoModelForCausalLM.from_pretrained(
    #"google/gemma-2b-it",
    model_name,
    device_map={"": Accelerator().local_process_index},
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=(
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    ),
)

tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer-chatml")
tokenizer.padding_side = "right"  # to prevent warnings

max_length = get_max_length(model)

dataset = preprocess_dataset(tokenizer, max_length, 1234, dataset)
eval_dataset = preprocess_dataset(tokenizer, max_length, 1234, eval_dataset)

#print(dataset[0])

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    tf32=True,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="none",
)

max_seq_length = 728

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    
print_trainable_parameters(model)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# Get lora module names
modules = find_all_linear_names(model)

"""
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=(
        LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=6,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    ),
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    dataset_text_field='text',
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
)
"""
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset = eval_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        warmup_steps=2,
        max_steps=10000,
        #optim="adamw_torch_fused",
        optim="paged_adamw_8bit",
        logging_steps=10,
        evaluation_strategy = 'steps',
        bf16=True,
        tf32=True,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none",
        eval_steps = 100, # should be 10 or 50
        save_steps = 100, # should be 10 or 50
        load_best_model_at_end=True,
    ),
    peft_config=(
        LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=6,
            bias="none",
            #target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    ),
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    dataset_text_field='text',
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
)
"""
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset = eval_dataset,
    args=TrainingArguments(
        report_to="none",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=1000, # should be 1000
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch_fused",
        #optim="paged_adamw_8bit",
        evaluation_strategy = 'steps',
        eval_steps = 10, # should be 10 or 50
        save_steps = 10, # should be 10 or 50
        load_best_model_at_end=True,
    ),
    peft_config=(
        LoraConfig(
            lora_alpha=8,
            lora_dropout=0.05,
            r=6,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    ),
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    dataset_text_field='text',
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
)
"""
trainer.train()

trainer.save_model()

