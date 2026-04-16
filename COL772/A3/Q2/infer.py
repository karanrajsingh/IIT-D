import sys
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"


torch.set_grad_enabled(False)

def normalize_label_prefix_flexible(noisy_label, labels):
    noisy_parts = noisy_label.strip("/").split("/")
    for label in labels:
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
    return text[:start] + f"<{tag}> {text[start:end]} </{tag}>" + text[end:]


def mark_both_entities(text, em1, em2):
    matches1 = list(re.finditer(re.escape(em1), text))
    matches2 = list(re.finditer(re.escape(em2), text))

    if not matches1 or not matches2:
        return text

    m1, m2 = min(
        [(m1, m2) for m1 in matches1 for m2 in matches2],
        key=lambda x: abs(x[0].start() - x[1].start())
    )

    if m1.start() > m2.start():
        first, second = (m1, "ent1"), (m2, "ent2")
    else:
        first, second = (m2, "ent2"), (m1, "ent1")

    text = (
        text[:first[0].start()]
        + f"<{first[1]}> {first[0].group()} </{first[1]}>"
        + text[first[0].end():]
    )
    text = mark_entity(text, second[0].group(), second[1])
    return text


def make_prompt(sentence, lang_tag):
    return f"{lang_tag} {sentence}\nrelation:"

def load_base_model():
    model_path = Path("/home/scai/msr/aiy247541/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323")
    if model_path.is_dir():
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
    else:
        print(f"Downloading model: {"Qwen_Qwen2.5-1.5B"}")
        model_path.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen_Qwen2.5-1.5B", device_map='cuda', torch_dtype=torch.float16)
        model.save_pretrained(str(model_path))
    return model

def load_model(output_dir, device):
    base_model_name = "Qwen/Qwen2.5-1.5B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # IMPORTANT: add same special tokens as training
    special_tokens = [
        "<ent1>", "</ent1>", "<ent2>", "</ent2>",
        "<en>", "<hi>", "<kn>", "<or>", "<tcy>"
    ]

    tokenizer.add_special_tokens({
        "bos_token": "<sos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "additional_special_tokens": special_tokens
    })

    base_model = load_base_model()

    base_model.resize_token_embeddings(len(tokenizer))

    finetune_path = Path(output_dir) / "finetune" / "classifier_model"

    model = PeftModel.from_pretrained(base_model, str(finetune_path))

    model.eval()

    model.to(device)

    return tokenizer, model


def clean_label(text):
    if text is None:
        return ""

    text = text.strip().lower()
    text = text.replace(" of ", "/")
    text = text.replace(" / ", "/")

    text = re.sub(r'[^a-z_/]', '', text)
    text = re.sub(r'/+', '/', text)

    return text.strip("/")


def split_parts(label):
    return [p for p in label.strip("/").split("/") if p]


def best_label_match(noisy_label, valid_labels):
    noisy_clean = clean_label(noisy_label)
    noisy_parts = split_parts(noisy_clean)

    if not noisy_parts:
        return None

    noisy_set = set(noisy_parts)

    best_label = None
    best_score = float("-inf")

    for label in valid_labels:
        label_parts = split_parts(label)
        label_set = set(label_parts)

        overlap = len(noisy_set & label_set)
        length_penalty = abs(len(label_parts) - len(noisy_parts)) * 0.2

        prefix_match = 0
        for i in range(min(len(noisy_parts), len(label_parts))):
            if noisy_parts[i] == label_parts[i]:
                prefix_match += 1
            else:
                break

        score = overlap + (0.5 * prefix_match) - length_penalty

        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def best_label_match_strict(noisy_label, valid_labels):
    noisy_clean = clean_label(noisy_label)

    for label in valid_labels:
        if noisy_clean == label.strip("/"):
            return label

    for label in valid_labels:
        if label.strip("/").startswith(noisy_clean):
            return label

    return best_label_match(noisy_label, valid_labels)


def generate_batch(model, tokenizer, prompts, device, valid_labels, max_new_tokens=20):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)


    labels = []
    for text in decoded:
        if "relation:" in text:
            label = text.split("relation:")[-1].strip()
        else:
            label = text.strip()

        label = label.split("\n")[0].strip()
        labels.append(best_label_match_strict(label, valid_labels))
    return labels


def main():
    lang = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = sys.argv[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_model(output_dir, device)

    dataset = load_dataset("json", data_files=test_file)["train"]

    lang_tag = f"<{lang}>"
    output_path = Path(output_dir) / f"Q2_{lang}.jsonl"

    base_path = Path(__file__).resolve().parent.parent

    label_map_path = base_path / 'sft_dataset' / 'hi_map.json'
    with open(label_map_path) as f:
        en_hi_dict = json.load(f)
    valid_labels = list(set(en_hi_dict.keys()))

    if lang == 'en':
        label_map_path = base_path / 'sft_dataset' / 'hi_map.json'
        with open(label_map_path) as f:
            en_lng_dict = json.load(f)
        lng_en_dict = {k:k for k in en_lng_dict.keys()}
        en_lng_dict = lng_en_dict
    elif lang == 'hi':
        label_map_path = base_path / 'sft_dataset' / 'hi_map.json'
        with open(label_map_path) as f:
            en_lng_dict = json.load(f)
        lng_en_dict = {v:k for k,v in en_lng_dict.items()}
    elif lang == 'kn':
        label_map_path = base_path / 'sft_dataset' / 'kn_map.json'
        with open(label_map_path) as f:
            en_lng_dict = json.load(f)
        lng_en_dict = {v:k for k,v in en_lng_dict.items()}
    elif lang == 'or':
        label_map_path = base_path / 'sft_dataset' / 'or_map.json'
        with open(label_map_path) as f:
            en_lng_dict = json.load(f)
        lng_en_dict = {v:k for k,v in en_lng_dict.items()}
    elif lang == 'tcy':
        label_map_path = base_path / 'sft_dataset' / 'tcy_map.json'
        with open(label_map_path) as f:
            en_lng_dict = json.load(f)
        lng_en_dict = {v:k for k,v in en_lng_dict.items()}

    valid_token_ids = tokenizer(valid_labels, add_special_tokens=False).input_ids

    jobs = []

    for sample_idx, sample in enumerate(dataset):
        sent = sample["sentText"]

        for rel_idx, rel in enumerate(sample["relationMentions"]):
            em1 = rel["em1Text"]
            em2 = rel["em2Text"]

            marked_sentence = mark_both_entities(sent, em1, em2)
            prompt = make_prompt(marked_sentence, lang_tag)

            jobs.append({
                "sample_idx": sample_idx,
                "rel_idx": rel_idx,
                "prompt": prompt,
                "em1": em1,
                "em2": em2
            })


    batch_size = 8
    results = {}

    for i in tqdm(range(0, len(jobs), batch_size), desc="Inference"):
        batch = jobs[i:i + batch_size]
        prompts = [job["prompt"] for job in batch]

        labels = generate_batch(model, tokenizer, prompts, device, valid_labels)

        for job, label in zip(batch, labels):
            key = (job["sample_idx"], job["rel_idx"])
            results[key] = {
                "em1Text": job["em1"],
                "em2Text": job["em2"],
                "label": en_lng_dict[label]
            }

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample_idx, sample in enumerate(dataset):

            predictions = []

            for rel_idx in range(len(sample["relationMentions"])):
                predictions.append(results[(sample_idx, rel_idx)])

            output_obj = {
                "articleId": sample.get("articleId",""),
                "sentId": sample.get("sentId",""),
                "sentText": sample.get("sentText",""),
                "relationMentions": predictions
            }

            fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
