# shell command to run this script:
#       accelerate launch --multi_gpu --num_machines 1 --num_processes 4 ppo.py

from huggingface_hub import hf_hub_download
import joblib

from huggingface_hub import login
login(token="your_hf_token")

import os
import re

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    data_file: Optional[str] = field(default="", metadata={"help": "the data file name in the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})

    
def setup_args(script_args:ScriptArguments):
    #script_args.model_name = "models/pm_sft_llama2/final_merged_checkpoint" # path to the final checkpoint of pretrained llama 2
    script_args.model_name = "gemma-2b-it-sft" 
    script_args.tokenizer_name = "gemma-2b-it-sft"
    script_args.reward_model_name = "models/reward_model_v2/peft_last_checkpoint"
    script_args.dataset_name = "Misaka19487/pennmutual" # insert ppo training dataset here.
    script_args.data_file = "ppo_data/train.csv"
    script_args.learning_rate = 1e-4 # same learning rate we used for sft and reward model
    script_args.max_length = 768 # should be 800
    script_args.output_max_length = 384 # should be 200
    script_args.mini_batch_size = 1 # (4)
    script_args.batch_size = 1 # (16) we were using 4 * 16. 4: number of gpus. 16 batchs each gpu (batch_size = mini_batch_size * gradient_accumulation_steps).
    script_args.ppo_epochs = 4 # (4) can be adjusted later
    script_args.gradient_accumulation_steps = 1 # (4)
    script_args.early_stopping = True
    script_args.target_kl = 0.1
    script_args.reward_baseline = -0.5
    script_args.batched_gen = False
    script_args.save_freq = 50
    script_args.output_dir = "models/gemma-2b-it-ppo_v2"
    script_args.seed = 1234
    
    return script_args

script_args = setup_args(ScriptArguments())

#parser = HfArgumentParser(ScriptArguments)
#script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)

# Below was an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.

# Modified dataset builder function specifically for pennmutual llama 2 with ppo project. 
# By Zhuoan. Feb 20 2024.
def build_dataset(
        tokenizer, dataset_name, data_file, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    
    data_files = {"train": data_file}
    train_dataset = load_dataset(dataset_name, data_files=data_files, use_auth_token=True, split="train")
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
            "reward_query": [],
        }
        for question, groundtruth in zip(examples["prompt"], examples["response"]):
            query = "Q: " + question + " Provide the name of the range at the end.\nA: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

            # add a query for reward model.
            #reward_query = "[score]Given the Answer and the Groundtruth, value the correctness. The correctness score is 0 or 1 and is to see if the final answer is correct.  Q: " + question + " Provide the name of the range at the end.\n Ground Truth: " + groundtruth + " A: "
            reward_query = "[score]Given the Answer and the Groundtruth, value the correctness. The correctness score is 0 or 1 and is to see if the final answer is correct.  Q: " + question + " Provide the name of the range at the end.\n A: "
            new_examples["reward_query"].append(reward_query)

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True
}


if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.tokenizer_name)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name, script_args.data_file)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32, # In sft, we used 64
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("column names of the dataset before dataloader:", dataset.column_names)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
 
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
reward_model = pipeline(
    "text-classification",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

print_first_batch = True

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
    
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    if print_first_batch:
        print(batch["response"])
        print_first_batch = False
        #break
    
    # Compute reward score
    #texts = ["[score]Given the Answer and the Groundtruth, value the correctness. The correctness score is 0 or 1 and is to see if the final answer is correct.  Q: " + q + " Provide the name of the range at the end.\n A: " + r for q, r in zip(batch["query"], batch["response"])]
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    reward_outputs = reward_model(texts, **rw_kwargs)
    # add function that extract the score from the reward model output here
    #rewards = [torch.tensor(float(re.search(r"Score: (\d+)", output).group(1)) - script_args.reward_baseline) for output in reward_outputs]
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in reward_outputs]
    
    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
        