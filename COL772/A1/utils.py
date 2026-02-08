import os
import re
from collections import Counter
import torch
import numpy as np
import random
import json
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained

stopwords = ['the', 'a', 'an' ,'.', 'of', 'and', 'at', 'on', 'by', 'with', 'for', 'from', 'up', 'or', 'but',
             'him', 'her', 'his', 'do', 'did']

# Decide on whether to keep punctuations and function words (Are they relevant to stylometry)
def clean_text(text):

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s!?;:"\'()-]', ' ', text) # keeps words and selected punctuation
    text = re.sub(r'([.,!?;:"\'()-])', r' \1 ', text)
    # text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()

    # tokens = [word for word in tokens if word not in stopwords]
    
    # counter = Counter(tokens)
    # print("Most common tokens:", counter.most_common(30))
    # print("Least common tokens:", counter.most_common(-30))

    # breakpoint()

    return tokens

def read_tokens_from_file(path_file):
    with open(path_file, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
        data = data.replace("\n", " ")
    cleaned_tokens = clean_text(data)
    return cleaned_tokens

def load_tokens(path_dir):
    print("Loading and cleaning data...")
    author_files = os.listdir(path_dir)
    token_list = []
    # breakpoint()
    for author_file in author_files:
        path_file = os.path.join(path_dir, author_file)
        token_file = read_tokens_from_file(path_file)
        token_list.append(token_file)
    # breakpoint()
    # Create train-test split here if needed
    return token_list

def build_vocabulary_indices(tokens_list, min_freq=2):
    print("Building vocabulary...")
    counter = Counter()
    for tokens in tokens_list:
        counter.update(tokens)
    
    vocab_words = []    
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab_words.append(word)
    # PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    # vocab = [PAD_TOKEN, UNK_TOKEN] + vocab_words
    vocab = [UNK_TOKEN] + vocab_words

    word_to_index = {}
    index_to_word = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index
    for word, index in word_to_index.items():
        index_to_word[index] = word
    return word_to_index, index_to_word, counter

def build_author_token_indices(tokens_list, word_to_index):
    print("Building document-wise token indices...")
    token_indices = []
    for tokens in tokens_list:
        index_tokens = []
        for token in tokens:
            if token in word_to_index:
                index_tokens.append(word_to_index[token])
            else:
                index_tokens.append(word_to_index["<UNK>"])
        token_indices.append(index_tokens)
    return token_indices

def build_contexts_target_pairs(token_indices, window=5, word_to_index=None):
    print("Building document-wise contexts-target pairs...")
    contexts_target_pairs = []
    for token_list in token_indices:
        for i in range(window, len(token_list) - window):
            if token_list[i] == word_to_index["<UNK>"]:
                continue
            context = []
            context.extend(token_list[i - window : i])
            context.extend(token_list[i+1 : i+window+1])
            target = token_list[i]
            contexts_target_pairs.append((context, target))
    print("Training data created!")
    return contexts_target_pairs

def build_distribution(word_to_index, counter, power=0.75):
    print("Building word distribution for negative sampling...")

    vocab_len = len(word_to_index)
    distribution = torch.zeros(vocab_len)
    for word, index in word_to_index.items():
        count = counter[word] 
        distribution[index] = np.pow(count, power)
    distribution = distribution / distribution.sum()
    return distribution

def train_val_split(token_list, split_ratio=0.8):
    print("Splitting data into training and validation sets...")
    train_token_list = []
    val_token_list = []
    for tokens in token_list:
        split_index = int(len(tokens)*split_ratio)
        train_token_list.append(tokens[:split_index])
        val_token_list.append(tokens[split_index:])
    return train_token_list, val_token_list

def apply_unk_dropout(token_indices, word_to_index, p_drop=0.1):
    unk_idx = word_to_index["<UNK>"]

    new_token_indices = []
    for token_list in token_indices:
        new_token_list = []
        for i, word_index in enumerate(token_list):
            if random.random() < p_drop:
                new_token_list.append(unk_idx)
            else:
                new_token_list.append(word_index)
        new_token_indices.append(new_token_list)
    
    return new_token_indices

def bigrams(tokens_list):
    bigrams_token_list = []
    for tokens in tokens_list:
        bigrams_tokens = []
        len_tokens = len(tokens)
        for i in range(len_tokens-1):
            bigrams_tokens.append(f"{tokens[i]} {tokens[i+1]}")
        bigrams_token_list.append(bigrams_tokens)
    return bigrams_token_list

def trigrams(tokens_list):
    bigrams_token_list = []
    for tokens in tokens_list:
        bigrams_tokens = []
        len_tokens = len(tokens)
        for i in range(len_tokens-2):
            bigrams_tokens.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        bigrams_token_list.append(bigrams_tokens)
    return bigrams_token_list

def task_1(path_file_in, word_to_index, embeddings, path_file_out):
    with open(path_file_in, "r", encoding="utf-8", errors="ignore") as f:
        data_list = json.load(f)
    out_list = []
    for data in data_list: 
        temp_dict = {}
        temp_dict["query_id"] = data["query_id"]
        query_tokens = clean_text(data["query_text"])
        query_indices = [word_to_index[token] for token in query_tokens if token in word_to_index]
        if len(query_indices) == 0:
            temp_dict["ranked_candidates"] = list(data["candidates"].keys())
            out_list.append(temp_dict)
            continue
        query_embedding = embeddings[query_indices].sum(dim=0).unsqueeze(0)
        rank_cand = {}
        for key in data["candidates"].keys():
            cand_tokens = clean_text(data["candidates"][key])
            cand_indices = [word_to_index[token] for token in cand_tokens if token in word_to_index]
            if len(cand_indices) == 0:
                rank_cand[key] = 0.0
                continue
            cand_embedding = embeddings[cand_indices].sum(dim=0).unsqueeze(0)
            rank_cand[key] = F.cosine_similarity(query_embedding, cand_embedding).item()
        temp_dict["ranked_candidates"] = [item[0] for item in sorted(rank_cand.items(), key=lambda item: item[1], reverse=True)]
        out_list.append(temp_dict)

    with open(path_file_out, "w", encoding="utf-8") as f:
        for item in out_list:
            f.write(json.dumps(item)+"\n")
    print("Task 1 output written!")
    return out_list

def task_2(path_file_in, word_to_index, embeddings, path_file_out):
    with open(path_file_in, "r", encoding="utf-8", errors="ignore") as f:
        data_list = json.load(f)
    k = data_list['num_authors']
    m = data_list['min_chunks_per_author']
    chunk_embeddings_list = []
    for chunk in data_list['chunks']:
        chunk_tokens = clean_text(chunk)
        chunk_indices = [word_to_index[token] for token in chunk_tokens if token in word_to_index]        
        if len(chunk_indices) == 0:
            empty_emb = torch.zeros(embeddings.size(1)).unsqueeze(0)
            chunk_embeddings_list.append(empty_emb)
            continue          
        chunk_embedding = embeddings[chunk_indices].sum(dim=0).unsqueeze(0)
        chunk_embeddings_list.append(chunk_embedding)
    chunk_embeddings = torch.cat(chunk_embeddings_list, dim=0).detach().numpy()
    kmeans = KMeansConstrained(n_clusters=k, size_min=m, random_state=0)
    cluster_labels = kmeans.fit_predict(chunk_embeddings)
    with open(path_file_out, "w", encoding="utf-8") as f:
        json.dump(cluster_labels.tolist(), f)
    print("Task 2 output written!")
    return cluster_labels.tolist()