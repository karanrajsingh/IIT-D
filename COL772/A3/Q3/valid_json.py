import json
from pathlib import Path


def convert_to_valid_json(input_filepath, output_filepath):
    """
    Reads a file containing concatenated/stacked multi-line JSON objects
    and saves them as a single valid JSON array.
    """
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    records = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        # 1. Skip whitespaces, newlines, or stray commas between objects
        while idx < len(content) and content[idx] in ' \t\n\r,':
            idx += 1
            
        if idx >= len(content):
            break
            
        # 2. Extract the next complete JSON object
        try:
            obj, end_idx = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end_idx # Move the index forward to the end of the parsed object
        except json.JSONDecodeError as e:
            print(f"Failed to decode at character index {idx}: {e}")
            break
    # 3. Write the extracted records into a new file as a proper JSON array
    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        # ensure_ascii=False keeps the Hindi characters intact instead of escaping them
        json.dump(records, f_out, ensure_ascii=False, indent=4)
    print(f"Success! Converted {len(records)} objects into a valid JSON array at '{output_filepath}'.")

def main():
    
    base_path = Path(__file__).resolve().parent.parent
    hi_train = base_path/'sft_dataset'/'hi_train.jsonl'
    kn_train = base_path/'sft_dataset'/'kn_train.jsonl'
    or_train = base_path/'sft_dataset'/'or_train.jsonl'
    tcy_train = base_path/'sft_dataset'/'tcy_val.jsonl'
    base_path = Path(__file__).resolve().parent
    base_dir = base_path/"cleaned_jsonl"
    base_dir.mkdir(exist_ok=True)
    hi_train_valid = base_dir/'hi_train_valid.json'
    kn_train_valid = base_dir/'kn_train_valid.json'
    or_train_valid = base_dir/'or_train_valid.json'
    tcy_train_valid = base_dir/'tcy_val_valid.json'
    for file, valid in zip([hi_train, kn_train, or_train, tcy_train], [hi_train_valid, kn_train_valid, or_train_valid, tcy_train_valid]):
        convert_to_valid_json(file, valid)
    print("All files formatted to json!")

if __name__ == "__main__":    main()

# import json
# def read_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # Strip whitespace and ignore empty lines
#             line = line.strip()
#             if line:
#                 # Parse the JSON string into a Python dictionary
#                 parsed_dict = json.loads(line)
#                 data.append(parsed_dict)
    
#     return data
# # Usage:
# # my_data = read_jsonl("my_file.jsonl")