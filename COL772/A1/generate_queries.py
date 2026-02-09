import json
import random
from pathlib import Path

data_dir = Path('data/train_data')
min_words, max_words = 250, 400
NUM_QUERIES = 1000
NUM_CHUNKS_PER_AUTHOR = 20

def load_author_texts():
    authors = {}
    for i in range(1, 11):
        path = data_dir / f'author_{i:03d}.txt'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # SPLIT INTO WORDS FIRST - Perfect boundaries
            words = full_text.split()
            chunks = []
            
            # Generate random word-based chunks
            for _ in range(NUM_CHUNKS_PER_AUTHOR * 2):
                start_word = random.randint(0, max(0, len(words) - max_words))
                end_word = random.randint(start_word + min_words, 
                                        min(start_word + max_words, len(words)))
                
                chunk_words = words[start_word:end_word]
                chunk = ' '.join(chunk_words)
                
                word_count = len(chunk_words)
                if min_words <= word_count <= max_words:
                    chunks.append(chunk)
            
            authors[f'auth_{i}'] = chunks[:NUM_CHUNKS_PER_AUTHOR]
    return authors

def sample_text(author_chunks):
    return random.choice(author_chunks)

# Generate dataset
author_texts = load_author_texts()
queries = []

print("Generating queries with COMPLETE WORDS...")
for qid in range(NUM_QUERIES):
    query_author_id = (qid % 10) + 1
    query_author = f'auth_{query_author_id}'
    query_text = sample_text(author_texts[query_author])
    
    candidates = {}
    for i in range(1, 11):
        candidates[f'cand_{i}'] = sample_text(author_texts[f'auth_{i}'])
    
    query = {
        "query_id": f"q_{qid}",
        "query_text": query_text,
        "candidates": candidates
    }
    queries.append(query)

# Save JSON
with open('queries.json', 'w', encoding='utf-8') as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

# print(f"✅ SUCCESS! Created queries.json with {NUM_QUERIES} queries")
# print("✅ 100% COMPLETE WORDS - NO CUT-OFFS!")
