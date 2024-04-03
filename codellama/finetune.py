import argparse
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
    AutoTokenizer
)
from datasets import IterableDataset
from trl import SFTTrainer
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('/home/mkbashar/Downloads/nlp/project/life2scenario_core')

from dataset_process import preprocess
from dataset_process import postprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="codellama/CodeLlama-13b-Instruct-hf")
    parser.add_argument("--dataset_name", type=str, default="life2scenario-llm24/Life2Scenario-medium")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument("--size_valid_set", type=int, default=2222)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--streaming", type=bool, default=False)
    parser.add_argument("--input_column_name", type=str, default="request")
    parser.add_argument("--output_column_name", type=str, default="response")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_seq_length", type=int, default=8000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_codellama")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_sample_text(example, input_column_name="request", output_column_name="response"):
    """Prepare the text from a sample of the dataset."""
    prompt = example[input_column_name]
    mod_prompt2 = prompt.split("```")[1].strip()
    prepro = preprocess.DataPreProcess(mod_prompt2)
    mod_prompt = prepro.data_preprocess()
    mod_prompt = prompt.split("```")[0].strip() + "\n```\n" + mod_prompt + "\n```"

    target = example[output_column_name]
    mod_target2 = target.split("```")[1].strip()
    postpro = preprocess.DataPreProcess(mod_target2)
    mod_target = postpro.data_preprocess()
    mod_target = target.split("```")[0].strip() + "\n```\n" + mod_target + "\n```"
    text = f"Question: {mod_prompt}\n\nAnswer: {mod_target}"
    return text

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        input_column_name="request",
        output_column_name="response"
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator), self.input_column_name, self.output_column_name))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }

def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="target", nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def gen():
    train = pd.read_csv('data/prep_pickles/train_dataset.csv')
    train = train[["request", "response"]]
    # return train
    for i in range(len(train)):
        prompt = train.iloc[i]["request"]
        target = train.iloc[i]["response"]
        yield {'request': prompt, 'response': target}

def create_datasets(tokenizer, args):
    train_data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split="train",
        token=args.token,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )

    valid_data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split="test",
        token=args.token,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    # dataset = IterableDataset.from_generator(gen)
    
    # print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")


    # if args.streaming:
    #     print("Loading the dataset in streaming mode")
    #     valid_data = dataset.take(args.size_valid_set)
    #     train_data = dataset.skip(args.size_valid_set)
    #     train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    # else:
    #     train_data = dataset["train"]
    #     valid_data = dataset["test"]
    #     print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )

    return train_dataset, valid_dataset

def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # load model and dataset
    args.token = os.environ.get("HF_TOKEN", None)
    if args.resume:
        args.model_id = "finetune_codellama2/checkpoint-300"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print_trainable_parameters(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    # setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=8000,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_steps=args.save_freq,
            eval_steps=args.eval_freq,
            save_strategy="steps",
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            logging_steps=args.log_freq,
            learning_rate=args.learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=args.max_steps,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="wandb",
            ddp_find_unused_parameters=False,
            output_dir=args.output_dir,
        ),
        packing=True,
        peft_config=lora_config,
        dataset_text_field=args.dataset_text_field,
    )

    print("Training...")
    if args.resume:
        print ("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=args.model_id)
    else:
        trainer.train()

    new_model = os.path.join(args.output_dir, "model")
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    if args.push_to_hub:
        trainer.push_to_hub("Upload model")
    print("Training Done! 💥")

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
