import torch
from unsupervised_corpus.unsupervised_corpus import ds_or, ds_hi, ds_kn, ds_tulu
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset
import re
from pathlib import Path
import json
import time
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F
import gc
from transformers import TrainerCallback, EarlyStoppingCallback
import random
import sys
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

TIMEOUT = 2350

class PretrainingDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ids = torch.tensor(self.data[index]['input_ids'])
        return {
            'input_ids': ids,
            'attention_mask': torch.tensor(self.data[index]['attention_mask']),
            'labels': ids.clone()
        }    

class TimeoutCallback(TrainerCallback):
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start_time > self.timeout_seconds:
            print(f"\nReached timeout of {self.timeout_seconds} seconds.")
            control.should_training_stop = True


def process_pretrain_datasets(tokenizer):

    def sample_data(ds, len, items=2200):
        return ds['train'].shuffle().select(range(min(items,len)))
    
    global ds_hi, ds_kn, ds_or, ds_tulu
    ds_hi_sampled = sample_data(ds_hi, len(ds_hi['train']))
    ds_kn_sampled = sample_data(ds_kn, len(ds_kn['train']))
    ds_or_sampled = sample_data(ds_or, len(ds_or['train']))
    ds_tcy_sampled = sample_data(ds_tulu, len(ds_tulu['train']))    
    
    def remove_english(datapoint):
        return {'text': re.sub(r"[A-Za-z]+", "", datapoint['text'])}

    def add_language_token(datapoint, language):
        return {'text': f"{language} " + datapoint['text']}
    ds_hi_sampled = ds_hi_sampled.map(remove_english)
    ds_hi_sampled = ds_hi_sampled.map(lambda x: add_language_token(x, '<hi>'))
    ds_kn_sampled = ds_kn_sampled.map(remove_english)
    ds_kn_sampled = ds_kn_sampled.map(lambda x: add_language_token(x, '<kn>'))
    ds_or_sampled = ds_or_sampled.map(remove_english)
    ds_or_sampled = ds_or_sampled.map(lambda x: add_language_token(x, '<or>'))
    ds_tcy_sampled = ds_tcy_sampled.map(remove_english)
    ds_tcy_sampled = ds_tcy_sampled.map(lambda x: add_language_token(x, '<tcy>'))

    ds_concat = concatenate_datasets([ds_hi_sampled, ds_kn_sampled, ds_or_sampled, ds_tcy_sampled])
    
    def tokenize_pretrain(datapoint):
        text = datapoint['text']
        tokens = tokenizer(text, padding='max_length', max_length=1024, truncation=True)
        return tokens

    def chunk_dataset(dataset, chunk_size=1024):
        all_input_ids = []
        all_attention_masks = []

        for example in dataset:
            ids = example['input_ids']
            mask = example['attention_mask']

            for i in range(0, len(ids), chunk_size):
                chunk_ids = ids[i:i+chunk_size]
                chunk_mask = mask[i:i+chunk_size]

                if len(chunk_ids) == chunk_size:
                    all_input_ids.append(chunk_ids)
                    all_attention_masks.append(chunk_mask)

        return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(all_input_ids, all_attention_masks)]
    
    ds_tokens = ds_concat.map(tokenize_pretrain)
    chunked_data = chunk_dataset(ds_tokens)
    random.shuffle(chunked_data)
    split_idx = int(len(chunked_data) * 0.9)
    train_chunks = chunked_data[:split_idx]
    val_chunks = chunked_data[split_idx:]
    train_dataset = PretrainingDataset(train_chunks)
    val_dataset = PretrainingDataset(val_chunks)
    return train_dataset, val_dataset
        
def load_model(model_name):
    model_path = Path("/home/scai/msr/aiy247541/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323")
    if model_path.is_dir():
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    else:
        print(f"Downloading model: {model_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', torch_dtype=torch.float16)
        model.save_pretrained(str(model_path))
    return model

def main():
    output_dir = Path(sys.argv[1])
    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    special_tokens = ["<ent1>", "</ent1>", "<ent2>", "</ent2>", "<en>", "<hi>", "<kn>", "<or>", "<tcy>"]
    tokenizer.add_special_tokens({"bos_token": "<sos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>", "additional_special_tokens": special_tokens})
    train_dataset, val_dataset = process_pretrain_datasets(tokenizer)
    t_tokenize = time.time()
    print(f'Time to prepare datasets: {t_tokenize - t_start:.2f} seconds')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_name = "Qwen/Qwen2.5-1.5B"
    base_model = load_model(model_name)
    base_model.resize_token_embeddings(len(tokenizer))

    for parameter in base_model.parameters():
        parameter.requires_grad =False
    
    adapter_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type='CAUSAL_LM')
    adapter_model = get_peft_model(base_model, adapter_config)

    adapter_model.to(device)

    base_path = output_dir
    base_adapter = base_path/'adapter'
    adapter_args = TrainingArguments(
        output_dir=str(base_adapter), 
        per_device_train_batch_size=2, 
        num_train_epochs=1, 
        learning_rate=2e-4,
        fp16=True,
        save_strategy="steps",            
        save_steps=100,
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
        gradient_accumulation_steps=4,
        logging_dir=f"{str(base_adapter)}/logs",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False)
    
    timeout_callback = TimeoutCallback(timeout_seconds=TIMEOUT) 

    adapter_trainer = Trainer(model=adapter_model, args=adapter_args, train_dataset=train_dataset, eval_dataset=val_dataset, callbacks=[EarlyStoppingCallback(early_stopping_patience=3), timeout_callback])
    adapter_trainer.train()    
    adapter_trainer.save_model(str(base_adapter/'pretrain_model'))

    t = time.time()

    print(f'Time taken: {t-t_start:.2f} seconds.')

if __name__ == "__main__":
    main()
