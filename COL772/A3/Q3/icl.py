import faiss
import sys
import json
import re
from typing import List
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
# from datasets import load_dataset
import time
from tqdm import tqdm

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return {"relationMentions": []}

def extract_first_relation(text):
    pattern = r'\n?\{\s*"relationMentions"\s*:\s*\[.*?\]\s*\}\n?'
    match = re.search(pattern, text, re.DOTALL)
    # breakpoint()
    data = json.loads(match.group(0))
    # breakpoint()
    return data

    # for rel in data.get("relationMentions", []):
    #     if rel.get("em1Text") == em1 and rel.get("em2Text") == em2:
    #        return rel

    # fallback
    # rels = data.get("relationMentions", [])
    # return rels[0] if rels else None



class FAISSRetriever:
    def __init__(self, texts):
        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBED_MODEL)

        print("Encoding corpus...")
        self.texts = texts
        embeddings = self.model.encode(texts, show_progress_bar=True)

        self.embeddings = np.array(embeddings).astype("float16")

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query, k=2):
        query_vec = self.model.encode([query]).astype("float16")
        distances, indices = self.index.search(query_vec, k)
        return [self.texts[i] for i in indices[0]]

def get_labels_from_examples(retrieved):
    labels = set()
    for ex in retrieved:
        matches = re.findall(r'"label":\s*"([^"]+)"', ex)
        labels.update(matches)
    return list(labels)

def build_prompt(test_ex, retriever, valid_labels):
    query = test_ex["sentText"] + " " + " ".join([rel["em1Text"] + " " + rel["em2Text"] for rel in test_ex["relationMentions"]])
    retrieved = retriever.retrieve(query, 2)
    subset_labels = list(set(get_labels_from_examples(retrieved)) & valid_labels)
    if len(subset_labels) == 0:
        subset_labels = list(valid_labels)
    label_str = ", ".join(subset_labels)
    # prompt = f"""Extract relations as JSON.\nReturn ONLY valid JSON.\nUse ONLY labels from: {label_str}\nFormat:\n{{"relationMentions":[{{"em1Text":"","em2Text":"","label":""}}]}}\n"""
    prompt = f"""You are a relation extraction system.\nExtract relations as JSON.\nReturn ONLY valid JSON.\nUse ONLY labels from: {label_str}\nFormat:\n{{"relationMentions":[{{"em1Text":"","em2Text":"","label":""}}]}}\nYour output should NEVER be {{"relationMentions":[]}}\nReturn output in least number of tokens\nReturn relationMentions strictly for the Input\n"""
    for ret in retrieved:
        prompt += f"\n### Example:\n{ret}\n"
    prompt += f"\n### Input\nSentence: {test_ex['sentText']}\n"
    prompt += "Entities: " + ", ".join([f"({rel['em1Text']}, {rel['em2Text']})" for rel in test_ex["relationMentions"]])
    prompt += "\n### Output:"
    return prompt

# def generate_response(llm, prompts):
#    sampling_params = SamplingParams(temperature=0.8,top_p=0.9,max_tokens=256)
#    outputs = llm.generate(prompts, sampling_params)
#    generated_texts = [output.outputs[0].text for output in outputs]
#    return generated_texts

def generate_response(llm, prompts, batch_size=16):
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=512
    )

    all_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
        batch_prompts = prompts[i:i + batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)
        # breakpoint()
        batch_texts = [output.outputs[0].text for output in outputs]
        all_outputs.extend(batch_texts)

    return all_outputs


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

def enforce_valid_labels(parsed, valid_labels):
    for rel in parsed.get("relationMentions", []):
        pred = rel.get("label", "")
        rel["label"] = normalize_label_strict(pred, valid_labels)
    return parsed

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

def normalize_label_strict(label, valid_labels):
    if label in valid_labels:
        return label
    for v in valid_labels:
        if label.strip() in v or v in label:
            return v
    return list(valid_labels)[0]

def main():

    t_start = time.time()

    la = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = sys.argv[3]

    lang = f'<{la}>'

    test_data = load_jsonl(test_file)
    base_path = Path(__file__).resolve().parent.parent
    base_dir = Path(__file__).resolve().parent
    base_dir = base_dir/"cleaned_jsonl"
    data = []
    valid_labels = set()
    if lang == '<en>':
        hi_label = base_path/'sft_dataset'/'hi_map.json'
        en_train = base_path/'en_sft_dataset'/'train.jsonl'
        en_train_ds = load_jsonl(str(en_train))
        with open(str(hi_label)) as file:
            en_hi_dict = json.load(file)
        valid_labels = set(en_hi_dict.keys())
        for i, datapoint in enumerate(en_train_ds):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                label = relation['label']
                if label not in en_hi_dict.keys():
                    label = normalize_label_prefix_flexible(label, en_hi_dict.keys())
                    if label is None:
                        print("Dropped label:", relation['label'])
                        continue
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                item = f"""\nSentence: {sentence}\nEntities:\n- ({em1Text}, {em2Text})\nOutput:\n{{"relationMentions": [{{"em1Text": "{em1Text}", "em2Text": "{em2Text}", "label": "{label}"}}]}}"""
                data.append(item)

    elif lang == '<hi>':
        hi_train = base_dir/'hi_train_valid.jsonl'
        hi_label = base_path/'sft_dataset'/'hi_map.json'
        hi_train_ds = load_jsonl(str(hi_train))
        with open(str(hi_label)) as file:
            en_hi_dict = json.load(file)
        valid_labels = set(en_hi_dict.values())
        for i, datapoint in enumerate(hi_train_ds):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                label_hi = relation['label']
                if label_hi not in en_hi_dict.values():
                    label_hi = normalize_label_prefix_flexible(label_hi, en_hi_dict.values())
                    if label_hi is None:
                        print("Dropped label:", relation['label'])
                        continue
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                item = f"""\nSentence: {sentence}\nEntities:\n- ({em1Text}, {em2Text})\nOutput:\n{{"relationMentions": [{{"em1Text": "{em1Text}", "em2Text": "{em2Text}", "label": "{label_hi}"}}]}}"""
                data.append(item)

    elif lang == '<kn>':
        kn_train = base_dir/'kn_train_valid.jsonl'
        kn_label = base_path/'sft_dataset'/'kn_map.json'
        kn_train_ds = load_jsonl(str(kn_train))
        with open(str(kn_label)) as file:
            en_kn_dict = json.load(file)
        valid_labels = set(en_kn_dict.values())
        for i, datapoint in enumerate(kn_train_ds):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                label_kn = relation['label']
                if label_kn not in en_kn_dict.values():
                    label_kn = normalize_label_prefix_flexible(label_kn, en_kn_dict.values())
                    if label_kn is None:
                        print("Dropped label:", relation['label'])
                        continue                
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                item = f"""\nSentence: {sentence}\nEntities:\n- ({em1Text}, {em2Text})\nOutput:\n{{"relationMentions": [{{"em1Text": "{em1Text}", "em2Text": "{em2Text}", "label": "{label_kn}"}}]}}"""
                data.append(item)

    elif lang == '<or>':
        or_train = base_dir/'or_train_valid.jsonl'
        or_label = base_path/'sft_dataset'/'or_map.json'
        or_train_ds = load_jsonl(str(or_train))
        with open(str(or_label)) as file:
            en_or_dict = json.load(file)
        valid_labels = set(en_or_dict.values())
        for i, datapoint in enumerate(or_train_ds):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                original_label = relation['label']
                label_or = original_label
                if label_or not in en_or_dict.values():
                    mapped = en_or_dict.get(original_label)
                    if mapped is not None:
                        label_or = mapped
                    else:
                        label_or = normalize_label_prefix_flexible(label_or, en_or_dict.values())
                        if label_or is None:
                            print("Dropped label:", original_label)
                            continue
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                item = f"""\nSentence: {sentence}\nEntities:\n- ({em1Text}, {em2Text})\nOutput:\n{{"relationMentions": [{{"em1Text": "{em1Text}", "em2Text": "{em2Text}", "label": "{label_or}"}}]}}"""
                data.append(item)

    elif lang == '<tcy>':     
        tcy_train = base_dir/'tcy_val_valid.jsonl'
        tcy_label = base_path/'sft_dataset'/'tcy_map.json'
        tcy_train_ds = load_jsonl(str(tcy_train))
        with open(str(tcy_label)) as file:
            en_tcy_dict = json.load(file)
        valid_labels = set(en_tcy_dict.values())
        for i, datapoint in enumerate(tcy_train_ds):
            base_sentence = datapoint["sentText"]
            for relation in datapoint['relationMentions']:
                em1Text = relation['em1Text']
                em2Text = relation['em2Text']
                original_label = relation['label']
                label_tcy = original_label
                if label_tcy not in en_tcy_dict.values():
                    mapped = en_tcy_dict.get(original_label)
                    if mapped is not None:
                        label_tcy = mapped
                    else:
                        label_tcy = normalize_label_prefix_flexible(label_tcy, en_tcy_dict.values())
                        if label_tcy is None:
                            print("Dropped label:", original_label)
                            continue
                sentence = mark_both_entities(base_sentence, em1Text, em2Text)
                item = f"""\nSentence: {sentence}\nEntities:\n- ({em1Text}, {em2Text})\nOutput:\n{{"relationMentions": [{{"em1Text": "{em1Text}", "em2Text": "{em2Text}", "label": "{label_tcy}"}}]}}"""
                data.append(item)

    retriever = FAISSRetriever(data)
    prompts = []
    for example in test_data:
        prompts.append(build_prompt(example, retriever, valid_labels))
    
    base_model_path = str(Path('/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659').resolve())
    llm = LLM(model=base_model_path, dtype='float16', max_model_len=4096)
    responses = generate_response(llm, prompts,batch_size=16)
    # breakpoint()
    outputs = []

    for example, response in zip(test_data, responses):
        parsed = extract_first_relation(response)
        # breakpoint()
        parsed = enforce_valid_labels(parsed, valid_labels)
        outputs.append({"articleId": example.get("articleId", ""), "sentId": example.get("sentId", ""), "sentText": example.get("sentText", ""), "relationMentions": parsed.get("relationMentions", [])})
    output_path = f"{output_dir}/output_{la}.jsonl"
    save_jsonl(outputs, output_path)

    t_end = time.time()
    output_path = f"{output_dir}/Q3_{la}.jsonl"
    output_path = f"{output_dir}/Q3_{la}.jsonl"
    save_jsonl(outputs, output_path)

    t_end = time.time()
    print(f"Total time taken: {t_end-t_start:.2f} seconds")

if __name__ == "__main__":
    main()
