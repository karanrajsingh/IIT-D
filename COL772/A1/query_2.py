import json
import random
from pathlib import Path
from typing import Dict, List

data_dir = Path('data/train_data')
NUM_AUTHORS = 10
MIN_CHUNKS_PER_AUTHOR = 25
TOTAL_CHUNKS = 350

def generate_clustering_dataset():
    authors_data: Dict[str, List[str]] = {}
    author_chunks: List[tuple] = []  # (chunk, author_id)
    
    # Load variable chunks per author (25-40)
    for i in range(1, NUM_AUTHORS + 1):
        path = data_dir / f'author_{i:03d}.txt'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            words = full_text.split()
            target_chunks = random.randint(MIN_CHUNKS_PER_AUTHOR, 40)
            author_chunks_list = []
            
            attempts = 0
            while len(author_chunks_list) < target_chunks and attempts < 100:
                start = random.randint(0, max(0, len(words) - 400))
                end = random.randint(start + 250, min(start + 400, len(words)))
                chunk = ' '.join(words[start:end])
                
                if 250 <= len(chunk.split()) <= 400 and chunk not in author_chunks_list:
                    author_chunks_list.append(chunk)
                attempts += 1
            
            authors_data[f'author_{i:03d}'] = author_chunks_list
            for chunk in author_chunks_list:
                author_chunks.append((chunk, i-1))  # 0-based author ID (0-9)
    
    # Shuffle for unlabeled dataset
    random.shuffle(author_chunks)
    unlabeled_chunks = [chunk for chunk, _ in author_chunks]
    true_labels = [author_id for _, author_id in author_chunks]
    
    dataset = {
        "num_authors": NUM_AUTHORS,
        "min_chunks_per_author": MIN_CHUNKS_PER_AUTHOR,
        "chunks": unlabeled_chunks[:TOTAL_CHUNKS]
    }
    
    return dataset, true_labels[:TOTAL_CHUNKS]

# Generate
dataset, true_labels = generate_clustering_dataset()

# Save unlabeled dataset (submit this)
with open('clustering_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

# Save JUST labels array
with open('true_labels.json', 'w', encoding='utf-8') as f:
    json.dump(true_labels, f, ensure_ascii=False)

print(f"✅ Created clustering_dataset.json ({len(dataset['chunks'])} chunks)")
print(f"✅ Created true_labels.json: {true_labels[:10]}... (length: {len(true_labels)})")
print(f"📊 k={dataset['num_authors']} authors, m≥{dataset['min_chunks_per_author']}")
