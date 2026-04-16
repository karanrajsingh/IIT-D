import json
from pathlib import Path

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main():
    
    base_path = Path(__file__).resolve().parent
    base_dir = base_path/"cleaned_jsonl"
    base_dir.mkdir(exist_ok=True)

    hi_train = base_dir/'hi_train_valid.json'
    kn_train = base_dir/'kn_train_valid.json'
    or_train = base_dir/'or_train_valid.json'
    tcy_train = base_dir/'tcy_val_valid.json'

    hi_trainl = base_dir/'hi_train_valid.jsonl'
    kn_trainl = base_dir/'kn_train_valid.jsonl'
    or_trainl = base_dir/'or_train_valid.jsonl'
    tcy_trainl = base_dir/'tcy_val_valid.jsonl'
    for file, valid in zip([hi_train, kn_train, or_train, tcy_train], [hi_trainl, kn_trainl, or_trainl, tcy_trainl]):
        convert_to_jsonl(file, valid)
    print("All jsons converted to jsonl!")

if __name__ == "__main__":    main()