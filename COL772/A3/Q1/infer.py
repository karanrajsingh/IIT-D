import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
import torch.nn as nn
import torch.nn.functional as F
import re
import sys
from tqdm import tqdm
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

# class ClassificationAttentionPoolModel(nn.Module):
#     def __init__(self, adapter_model, num_labels, dropout=0.1):
#         super().__init__()
#         self.base = adapter_model
#         hidden = adapter_model.config.hidden_size
#         self.attention = nn.Linear(hidden, 1)
#         self.dropout = nn.Dropout(p=dropout)
#         self.classifier = nn.Linear(hidden, num_labels)

#     def attention_pool(self, hidden_states, attention_mask):
#         scores = self.attention(hidden_states).squeeze(-1)
#         scores = scores.masked_fill(attention_mask == 0, float("-inf"))
#         weights = F.softmax(scores, dim=1)
#         pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
#         return pooled

#     def forward(self, input_ids, attention_mask):
#         outputs = self.base(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True
#         )
#         hidden_states = outputs.hidden_states[-1]
#         pooled = self.attention_pool(hidden_states, attention_mask)
#         pooled = self.dropout(pooled)
#         logits = self.classifier(pooled)
#         return logits

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

    def forward(self, input_ids, attention_mask):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        pooled = self.mean_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits

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
    return text[:start] + f"<{tag}> {text[start:end]} </{tag}>" + text[end:]


# def mark_both_entities(text, em1, em2):
#     matches1 = list(re.finditer(re.escape(em1), text))
#     matches2 = list(re.finditer(re.escape(em2), text))

#     if not matches1 or not matches2:
#         return text

#     m1, m2 = min(
#         [(m1, m2) for m1 in matches1 for m2 in matches2],
#         key=lambda x: abs(x[0].start() - x[1].start())
#     )

#     if m1.start() > m2.start():
#         first, second = (m1, "ent1"), (m2, "ent2")
#     else:
#         first, second = (m2, "ent2"), (m1, "ent1")

#     text = (
#         text[:first[0].start()]
#         + f"<{first[1]}> {first[0].group()} </{first[1]}>"
#         + text[first[0].end():]
#     )

#     text = mark_entity(text, second[0].group(), second[1])
#     return text

def mark_both_entities(text, em1, em2):
    # mark ent1 first occurrence
    text, n1 = re.subn(re.escape(em1), f"<ent1> {em1} </ent1>", text, count=1)

    # mark ent2 first occurrence (on updated text)
    text, n2 = re.subn(re.escape(em2), f"<ent2> {em2} </ent2>", text, count=1)

    return text


def load_base_model():
    model_path = Path("/home/scai/msr/aiy247541/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323")
    # breakpoint()
    if model_path.is_dir():
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    else:
        print(f"Downloading model: {"Qwen_Qwen2.5-1.5B"}")
        model_path.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen_Qwen2.5-1.5B", device_map='cuda', torch_dtype=torch.float16)
        model.save_pretrained(str(model_path))
    return model


def load_model(output_dir, device):
    base_path = Path(output_dir)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    special_tokens = ["<ent1>", "</ent1>", "<ent2>", "</ent2>", "<en>", "<hi>", "<kn>", "<or>", "<tcy>"]
    tokenizer.add_special_tokens({
        "bos_token": "<sos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": special_tokens
    })


    base_model = load_base_model()
    base_model.resize_token_embeddings(len(tokenizer))

    adapter_model = PeftModel.from_pretrained(base_model, str(base_path / "adapter" / "pretrain_model"))
    # adapter_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type='CAUSAL_LM', lora_dropout=0.1)
    # adapter_model = get_peft_model(base_model, adapter_config)
    adapter_model.resize_token_embeddings(len(tokenizer))

    with open(base_path / "dictionary.json", "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    en_id_to_label =  dictionary['en_id_to_label']
    en_hi_dict = dictionary['en_hi_dict']
    en_kn_dict = dictionary['en_kn_dict']

    model = ClassificationAttentionPoolModel(adapter_model, len(en_id_to_label))

    model.load_state_dict(torch.load(base_path / "finetune" / "classifier_model" / "pytorch_model.bin"))
    model.to(device)
    model.eval()

    return model, tokenizer, en_id_to_label, en_hi_dict, en_kn_dict

def infer(test_file, output_dir, lang):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, en_id_to_label, en_hi_dict, en_kn_dict = load_model(output_dir, device)

    if lang == "hi":
        final = en_hi_dict
    elif lang == "kn":
        final = en_kn_dict
    elif lang == "en":
        final = {l : l for l in en_id_to_label.values()}

    with open(test_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    jobs = []

    for sample_idx, item in enumerate(data):
        sent = item["sentText"]

        for rel_idx, rel in enumerate(item["relationMentions"]):
            em1 = rel["em1Text"]
            em2 = rel["em2Text"]

            sentence = mark_both_entities(sent, em1, em2)
            sentence = f"<{lang}> " + sentence

            jobs.append({
                "sample_idx": sample_idx,
                "rel_idx": rel_idx,
                "sentence": sentence,
                "em1": em1,
                "em2": em2
            })

    batch_size = 8 # 16 works, but 8 for safe side
    results = {}

    for i in tqdm(range(0, len(jobs), batch_size), desc="Inference"):
        batch = jobs[i:i + batch_size]
        sentences = [job["sentence"] for job in batch]

        tokens = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1152
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = model(tokens["input_ids"], tokens["attention_mask"])
            preds = torch.argmax(logits, dim=-1).tolist()

        for job, pred in zip(batch, preds):
            key = (job["sample_idx"], job["rel_idx"])
   
            results[key] = {
                "em1Text": job["em1"],
                "em2Text": job["em2"],
                "label": final[en_id_to_label[str(pred)]]
            }

    output = []

    for sample_idx, item in enumerate(data):
        relations_out = []

        for rel_idx in range(len(item["relationMentions"])):
            relations_out.append(results[(sample_idx, rel_idx)])

        output.append({
            "articleId": item.get("articleId", ""),
            "sentId": item.get("sentId", ""),
            "sentText": item["sentText"],
            "relationMentions": relations_out
        })

    out_path = Path(output_dir) / f"Q1_{lang}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for o in output:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    lang = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = sys.argv[3]

    infer(test_file, output_dir, lang)
