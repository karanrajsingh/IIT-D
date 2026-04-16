import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import re
from pathlib import Path
import json
import time
from peft import LoraConfig, PeftModel, get_peft_model
import torch.nn as nn
import torch.nn.functional as F
import gc
from transformers import TrainerCallback, EarlyStoppingCallback, DataCollatorWithPadding
import random
import sys
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

TIMEOUT = 10000

class FinetuningDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        return {
            # 'input_ids': torch.tensor(self.data[index]['input_ids']),
            # 'attention_mask': torch.tensor(self.data[index]['attention_mask']),
            # 'labels': torch.tensor(self.data[index]['label_id'])
            'input_ids': self.data[index]['input_ids'],
            'attention_mask': self.data[index]['attention_mask'],
            'labels': self.data[index]['label_id']
        }
    

# class ClassificationAttentionPoolModel(nn.Module):
#     def __init__(self, adapter_model, num_labels, dropout=0.1):
#         super().__init__()
#         self.base = adapter_model
#         hidden = adapter_model.config.hidden_size
#         self.attention = nn.Linear(hidden, 1)
#         self.dropout = nn.Dropout(p=dropout)
#         self.classifier = nn.Linear(hidden, num_labels)
#         self.loss_fn = nn.CrossEntropyLoss()

#     def attention_pool(self, hidden_states, attention_mask):
#         scores = self.attention(hidden_states).squeeze(-1)
#         scores = scores.masked_fill(attention_mask == 0, float("-inf"))
#         weights = F.softmax(scores, dim=1)
#         weights = weights.unsqueeze(-1)
#         pooled = torch.sum(hidden_states*weights, dim=1)
#         return pooled

#     def forward(self, input_ids, attention_mask,labels):
#         outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
#         hidden_states = outputs.hidden_states[-1]
#         pooled = self.attention_pool(hidden_states, attention_mask)
#         pooled = self.dropout(pooled)
#         logits = self.classifier(pooled)
#         loss = self.loss_fn(logits, labels)
#         return {"loss": loss, "logits": logits}


class ClassificationAttentionPoolModel(nn.Module):
    def __init__(self, adapter_model, num_labels, dropout=0.1):
        super().__init__()
        self.base = adapter_model
        hidden = adapter_model.config.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        pooled = self.mean_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

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

def normalize_label_prefix_flexible(noisy_label, label_dict):
    noisy_parts = noisy_label.strip("/").split("/")
    for label in label_dict:
        parts = label.strip("/").split("/")
        if len(parts) < len(noisy_parts):
            continue
        match = True
        for i in range(len(noisy_parts)):
            if i == len(noisy_parts) - 1:
                if not parts[i].startswith(noisy_parts[i]):
                    match = False
                    break
            else:
                if parts[i] != noisy_parts[i]:
                    match = False
                    break
        if match:
            return label
    return None

def mark_entity(text, entity, tag, other_entity=None):
    matches = [m for m in re.finditer(re.escape(entity), text)]
    if not matches:
        return text
    if other_entity is None:
        match = matches[0]
    else:
        other_match = re.search(re.escape(other_entity), text)            
        if other_match is None:
            match = matches[0]
        else:
            other_pos = other_match.start()
            match = min(matches, key=lambda m: abs(m.start() - other_pos))
    start, end = match.start(), match.end()
    return (
        text[:start]
        + f"<{tag}> {text[start:end]} </{tag}>"
        + text[end:]
    )

def mark_both_entities(text, em1, em2):
    matches1 = list(re.finditer(re.escape(em1), text))
    matches2 = list(re.finditer(re.escape(em2), text))

    if not matches1 or not matches2:
        return text
    m1, m2 = min([(m1, m2) for m1 in matches1 for m2 in matches2], key=lambda x: abs(x[0].start() - x[1].start()))

    if m1.start() > m2.start():
        first, second = (m1, "ent1"), (m2, "ent2")
    else:
        first, second = (m2, "ent2"), (m1, "ent1")

    text = (text[:first[0].start()] + f"<{first[1]}> {first[0].group()} </{first[1]}>" + text[first[0].end():])
    text = mark_entity(text, second[0].group(), second[1])
    return text

# def mark_both_entities(text, em1, em2):
#     # mark ent1 first occurrence
#     text, n1 = re.subn(re.escape(em1), f"<ent1> {em1} </ent1>", text, count=1)

#     # mark ent2 first occurrence (on updated text)
#     text, n2 = re.subn(re.escape(em2), f"<ent2> {em2} </ent2>", text, count=1)

#     return text

def process_fine_datasets(tokenizer):

    # first = 0.8
    # second = 0.7
    first = 1
    second = 0.9

    base_path = Path(__file__).resolve().parent.parent
    def tokenize_finetune(sentence):
        # tokens = tokenizer(sentence, padding='max_length', max_length=1152, truncation=True)
        tokens = tokenizer(sentence, max_length=1152, truncation=True)
        return tokens
    
    def add_language_token_finetune(datapoint, language):
        return f"{language} " + datapoint
    
    offset = 0
    id_to_label = {}
    ds_finetune = []   
    ds_finetune_val = [] 

    # English finetune
    hi_label = base_path/'sft_dataset'/'hi_map.json'
    en_train = base_path/'en_sft_dataset'/'train.jsonl'
    en_train_ds = load_dataset("json", data_files=str(en_train))    
    with open(str(hi_label)) as file:
        en_hi_dict = json.load(file)
    en_label_to_id = {label: id + offset for id, label in enumerate(en_hi_dict.keys())}
    en_id_to_label = {id + offset: label for id, label in enumerate(en_hi_dict.keys())}
    # offset += len(en_hi_dict.keys())
    for i, datapoint in enumerate(en_train_ds['train'].select(range(int(first*len(en_train_ds['train']))))):
        base_sentence = datapoint["sentText"]
        for relation in datapoint['relationMentions']:
            em1Text = relation['em1Text']
            em2Text = relation['em2Text']
            label = relation['label']
            label_id = en_label_to_id.get(label, -1)
            if label_id == -1:
                label = normalize_label_prefix_flexible(label, en_label_to_id)
                if label is None:
                    print("Dropped label:", relation['label'])
                    continue
                label_id = en_label_to_id[label]
            sentence = mark_both_entities(base_sentence, em1Text, em2Text)
            sentence = add_language_token_finetune(sentence, '<en>')
            sentence_tokens = tokenize_finetune(sentence)
            if i < int(second*len(en_train_ds['train'])):
                ds_finetune.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
            else:
                ds_finetune_val.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
    id_to_label.update(en_id_to_label)


    # Hindi finetune
    hi_train = base_path/'sft_dataset'/'hi_train.jsonl'
    hi_label = base_path/'sft_dataset'/'hi_map.json'
    hi_train_ds = load_dataset("json", data_files=str(hi_train))
    with open(str(hi_label)) as file:
        en_hi_dict = json.load(file)
    hi_en_dict = {v: k for k, v in en_hi_dict.items()}
    # hi_label_to_id = {label: id + offset for id, label in enumerate(en_hi_dict.values())}
    # hi_id_to_label = {id + offset: label for id, label in enumerate(en_hi_dict.values())}
    # offset += len(en_hi_dict.keys())
    for i, datapoint in enumerate(hi_train_ds['train'].select(range(int(first*len(hi_train_ds['train']))))):
        base_sentence = datapoint["sentText"]
        for relation in datapoint['relationMentions']:
            em1Text = relation['em1Text']
            em2Text = relation['em2Text']
            label_hi = relation['label']
            label_en = hi_en_dict.get(label_hi, None)
            # label_id = hi_label_to_id.get(label_hi, -1)
            if label_en is None:
                label_hi = normalize_label_prefix_flexible(label_hi, hi_en_dict)
                if label_hi is None:
                    print("Dropped label:", relation['label'])
                    continue
                label_en = hi_en_dict[label_hi]
                label_id = en_label_to_id[label_en]
            sentence = mark_both_entities(base_sentence, em1Text, em2Text)
            sentence = add_language_token_finetune(sentence, '<hi>')
            sentence_tokens = tokenize_finetune(sentence)
            if i < int(second*len(hi_train_ds['train'])):
                ds_finetune.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
            else:
                ds_finetune_val.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})

    # Kanada finetune
    kn_train = base_path/'sft_dataset'/'kn_train.jsonl'
    kn_label = base_path/'sft_dataset'/'kn_map.json'
    kn_train_ds = load_dataset("json", data_files=str(kn_train))
    with open(str(kn_label)) as file:
        en_kn_dict = json.load(file)
    kn_en_dict = {v: k for k, v in en_kn_dict.items()}
    # kn_label_to_id = {label: id + offset for id, label in enumerate(en_kn_dict.values())}
    # kn_id_to_label = {id + offset: label for id, label in enumerate(en_kn_dict.values())}
    # offset += len(en_kn_dict.keys())
    for i, datapoint in enumerate(kn_train_ds['train'].select(range(int(first*len(kn_train_ds['train']))))):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                label_kn = relation['label']
                label_en = kn_en_dict.get(label_kn, None)
                # label_id = kn_label_to_id.get(label_kn, -1)
                if label_en is None:
                    label_kn = normalize_label_prefix_flexible(label_kn, kn_en_dict)
                    if label_kn is None:
                        print("Dropped label:", relation['label'])
                        continue
                    label_en = kn_en_dict[label_kn]
                    label_id = en_label_to_id[label_en]
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                sentence = add_language_token_finetune(sentence, '<kn>')
                sentence_tokens = tokenize_finetune(sentence)
                if i < int(second*len(kn_train_ds['train'])):
                    ds_finetune.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
                else:
                    ds_finetune_val.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})

    # # Oriya finetune
    # or_train = base_path/'sft_dataset'/'or_train.jsonl'
    # or_label = base_path/'sft_dataset'/'or_map.json'
    # or_train_ds = load_dataset("json", data_files=str(or_train))
    # with open(str(or_label)) as file:
    #     en_or_dict = json.load(file)
    # or_label_to_id = {label: id + offset for id, label in enumerate(en_or_dict.values())}
    # or_id_to_label = {id + offset: label for id, label in enumerate(en_or_dict.values())}
    # offset += len(en_or_dict.keys())
    # for i, datapoint in enumerate(or_train_ds['train'].select(range(int(first*len(or_train_ds['train']))))):
    #         base_sentence = datapoint["sentText"]
    #         for relation in datapoint['relationMentions']:
    #             em1Text = relation['em1Text']
    #             em2Text = relation['em2Text']
    #             # label_or = relation['label']
    #             # label_id = or_label_to_id.get(label_or, -1)
    #             # if label_id == -1:
    #             #     mapped = en_or_dict.get(relation['label'], None)
    #             #     if mapped is not None:
    #             #         label_or = en_or_dict.get(relation['label'], None)
    #             #     else:
    #             #         label_or = normalize_label_prefix_flexible(label_or, or_label_to_id)
    #             #         if label_or is None:
    #             #             print("Dropped label:", relation['label'])
    #             #             continue
    #             #     label_id = or_label_to_id[label_or]
    #             #     if label_id == -1:
    #             #         print("Still invalid after normalization:", relation['label'])
    #             #         continue
    #             original_label = relation['label']
    #             label_or = original_label
    #             label_id = or_label_to_id.get(label_or, -1)
    #             if label_id == -1:
    #                 mapped = en_or_dict.get(original_label)
    #                 if mapped is not None:
    #                     label_or = mapped
    #                 else:
    #                     label_or = normalize_label_prefix_flexible(label_or, or_label_to_id)
    #                     if label_or is None:
    #                         print("Dropped label:", original_label)
    #                         continue
    #                 label_id = or_label_to_id.get(label_or, -1)
    #                 if label_id == -1:
    #                     print("Still invalid after normalization:", original_label)
    #                     continue
    #             sentence = mark_both_entities(base_sentence, em1Text, em2Text)
    #             sentence = add_language_token_finetune(sentence, '<or>')
    #             sentence_tokens = tokenize_finetune(sentence)
    #             if i < int(second*len(or_train_ds['train'])):
    #                 ds_finetune.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
    #             else:
    #                 ds_finetune_val.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
    # id_to_label.update(or_id_to_label)


    # # Tulu finetune
    # tcy_train = base_path/'sft_dataset'/'tcy_val.jsonl'
    # tcy_label = base_path/'sft_dataset'/'tcy_map.json'
    # tcy_train_ds = load_dataset("json", data_files=str(tcy_train))
    # with open(str(tcy_label)) as file:
    #     en_tcy_dict = json.load(file)
    # tcy_label_to_id = {label: id + offset for id, label in enumerate(en_tcy_dict.values())}
    # tcy_id_to_label = {id + offset: label for id, label in enumerate(en_tcy_dict.values())}
    # for i, datapoint in enumerate(tcy_train_ds['train'].select(range(int(first*len(tcy_train_ds['train']))))):
    #         base_sentence = datapoint["sentText"]
    #         for relation in datapoint['relationMentions']:
    #             em1Text = relation['em1Text']
    #             em2Text = relation['em2Text']
    #             # label_tcy = relation['label']
    #             # label_id = tcy_label_to_id.get(label_tcy, -1)
    #             # if label_id == -1:
    #             #     label_tcy = normalize_label_prefix_flexible(label_tcy, tcy_label_to_id)
    #             #     if label_tcy is None:
    #             #         print("Dropped label:", relation['label'])
    #             #         continue
    #             #     label_id = tcy_label_to_id[label_tcy]
    #             original_label = relation['label']
    #             label_or = original_label
    #             label_id = or_label_to_id.get(label_or, -1)
    #             if label_id == -1:
    #                 mapped = en_or_dict.get(original_label)
    #                 if mapped is not None:
    #                     label_or = mapped
    #                 else:
    #                     label_or = normalize_label_prefix_flexible(label_or, or_label_to_id)
    #                     if label_or is None:
    #                         print("Dropped label:", original_label)
    #                         continue
    #                 label_id = or_label_to_id.get(label_or, -1)
    #                 if label_id == -1:
    #                     print("Still invalid after normalization:", original_label)
    #                     continue
    #             sentence = mark_both_entities(base_sentence, em1Text, em2Text)
    #             sentence = add_language_token_finetune(sentence, '<tcy>')
    #             sentence_tokens = tokenize_finetune(sentence)
    #             if i < int(second*len(tcy_train_ds['train'])):
    #                 ds_finetune.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
    #             else:
    #                 ds_finetune_val.append({'input_ids': sentence_tokens['input_ids'], 'attention_mask': sentence_tokens['attention_mask'], 'label_id': label_id})
    # id_to_label.update(tcy_id_to_label)

    random.shuffle(ds_finetune)
    random.shuffle(ds_finetune_val)
    fine_tuning_dataset = FinetuningDataset(ds_finetune)
    fine_tuning_dataset_val = FinetuningDataset(ds_finetune_val)
    return fine_tuning_dataset, fine_tuning_dataset_val, en_id_to_label, en_hi_dict, en_kn_dict
        
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
    output_dir.mkdir(exist_ok=True)
    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    special_tokens = ["<ent1>", "</ent1>", "<ent2>", "</ent2>", "<en>", "<hi>", "<kn>", "<or>", "<tcy>"]
    tokenizer.add_special_tokens({"bos_token": "<sos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>", "additional_special_tokens": special_tokens})
    fine_tuning_dataset, fine_tuning_dataset_val,  en_id_to_label, en_hi_dict, en_kn_dict = process_fine_datasets(tokenizer)
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
    
    adapter_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type='CAUSAL_LM', lora_dropout=0.1)
    # adapter_model = get_peft_model(base_model, adapter_config)
    # load pretrained model weights
    base_path = output_dir
    base_adapter = base_path/'adapter'
    adapter_model = PeftModel.from_pretrained(base_model, str(base_adapter/'pretrain_model'))
    # adapter_model = get_peft_model(base_model, adapter_config)
    adapter_model.resize_token_embeddings(len(tokenizer))
    adapter_model.to(device)

    num_labels = len(en_id_to_label.values())
    finetune_model = ClassificationAttentionPoolModel(adapter_model, num_labels)
    finetune_model.to(device)

    base_path = output_dir
    finetune_dir = base_path/'finetune'
    finetune_args = TrainingArguments(
        output_dir=str(finetune_dir),
        per_device_train_batch_size=6,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="steps",            
        save_steps=500,
        save_total_limit=20,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=False,
        logging_dir=f"{str(finetune_dir)}/logs",
        logging_steps=10,
        gradient_accumulation_steps=6,
        logging_first_step=True,
        disable_tqdm=False)
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    timeout_callback = TimeoutCallback(timeout_seconds=TIMEOUT) 

    finetune_trainer = Trainer(model=finetune_model, args=finetune_args, train_dataset=fine_tuning_dataset, eval_dataset=fine_tuning_dataset_val, callbacks=[EarlyStoppingCallback(early_stopping_patience=3), timeout_callback], data_collator=data_collator)
    finetune_trainer.train()
    finetune_trainer.save_model(str(finetune_dir/ "classifier_model"))

    save_dict_path = output_dir
    save_dict_path.mkdir(exist_ok=True)
    with open(f"{str(save_dict_path)}/dictionary.json", "w", encoding="utf-8") as f:
        json.dump({
            "en_id_to_label": en_id_to_label,
            "en_hi_dict": en_hi_dict,
            "en_kn_dict": en_kn_dict
        }, f, ensure_ascii=False, indent=2)
    t = time.time()

    print(f'Time taken: {t-t_start:.2f} seconds.')

if __name__ == "__main__":
    main()
