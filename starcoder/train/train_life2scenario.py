# %%
import os
from os import path
import pandas as pd
import numpy as np
import glob

# %%
import torch
from finetune import create_datasets, ConstantLengthDataset, chars_token_ratio, run_training, prepare_model_for_int8_training, print_trainable_parameters
from finetune import SavePeftModelCallback, LoadBestPeftModelCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# %%
from datasets import load_dataset
from datasets import Dataset, Features, Value


from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# %% [markdown]
# # Definitions

# %%
HOME=os.path.expanduser('~')
LIFE2SCENARIO_ROOT_PATH=path.join(HOME,"Documents/life2scenario/")
DATASET_ROOT_PATH=path.join(LIFE2SCENARIO_ROOT_PATH,"life2scenario_minimal/dataset/train/")

print(DATASET_ROOT_PATH)

# %%
PROMPTS_ROOT=path.join(DATASET_ROOT_PATH, "prompts")
REFERENCE_ROOT=path.join(DATASET_ROOT_PATH, "ref_scenarios")
TARGET_ROOT=path.join(DATASET_ROOT_PATH, "target_scenarios")

# %%
PREP_PICKLES_ROOT=path.join(LIFE2SCENARIO_ROOT_PATH, "prep_pickles")

# %% [markdown]
# ## Load Train DataFrame

# %%
train_final = pd.read_csv(path.join(PREP_PICKLES_ROOT, "train_dataset.csv"))

train_final = train_final[["request", "response"]]
train_final.head()

# %% [markdown]
# ## Create Dataset

# %%
life2scenario_dataset = Dataset.from_pandas(
  train_final,
  features=Features(
    {'request': Value('string'),
     'response': Value('string')
    })
)


l2s_dataset = life2scenario_dataset.train_test_split(test_size=0.1)
l2s_dataset

# %%
class Dict2Obj(object):
  def __init__(self, dictionary):
    for key in dictionary:
        setattr(self, key, dictionary[key])
  
  def __repr__(self):
    return "<dict2obj: %s>" % self.__dict__

# Training Params
train_dict = {
    "model_path": "bigcode/starcoderbase-1b",
    "subset": "data/finetune",
    "streaming": True,
    "seq_length": 8000,
    "max_steps": 1000,
    "batch_size": 2,
    "input_column_name": "request",
    "output_column_name": "response",
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 100,
    "weight_decay": 0.05,
    "output_dir": "./checkpoints_v2",

    "local_rank": 0,
    "eos_token_id": 49152,
    "no_gradient_checkpointing": False,
    "shuffle_buffer": 5000,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "no_fp16": False,
    "bf16":False,
    "seed": 0,
    "num_workers": 32,
    "log_freq": 1,
    "eval_freq":10,
    "save_freq": 10
  }

train_args = Dict2Obj(train_dict)
train_args

# %%
checkpoint = "bigcode/starcoderbase-1b"
device = "cuda" # for GPU usage or "cpu" for CPU usage

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

tokenizer = AutoTokenizer.from_pretrained(checkpoint, load_in_8bit=True,
 device_map={"": Accelerator().process_index})
# to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
        use_auth_token=True,
        use_cache=not train_args.no_gradient_checkpointing,
        load_in_8bit=True,
        device_map={"": Accelerator().process_index},
        max_memory=get_gpus_max_memory("70GB"))

# %%
model.num_parameters()

# %%
hf_train_data = l2s_dataset["train"]
hf_test_data = l2s_dataset["test"]


chars_per_token = chars_token_ratio(hf_train_data, tokenizer, train_args.input_column_name, train_args.output_column_name)
print(f"chars_per_token: {chars_per_token}")

train_dataset = ConstantLengthDataset(
    tokenizer,
    hf_train_data,
    infinite=True,
    seq_length=train_args.seq_length,
    chars_per_token=chars_per_token,
    input_column_name=train_args.input_column_name,
    output_column_name=train_args.output_column_name
)

valid_dataset = ConstantLengthDataset(
    tokenizer,
    hf_test_data,
    infinite=False,
    seq_length=train_args.seq_length,
    chars_per_token=chars_per_token,
    input_column_name=train_args.input_column_name,
    output_column_name=train_args.output_column_name
)


def run_training(model, args, train_data, val_data):
    print("Loading the model")
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_proj", "c_attn", "q_attn"]
    )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback])

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


run_training(model=model, args=train_args, train_data=train_dataset, val_data=valid_dataset)