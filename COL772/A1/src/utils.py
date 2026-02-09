import os
import re
from collections import Counter
import torch
import numpy as np
import random
import json
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time

# stopwords = ['the', 'a', 'an' ,'.', 'of', 'and', 'at', 'on', 'by', 'with', 'for', 'from', 'up', 'or', 'but',
#              'him', 'her', 'his', 'do', 'did']
def clean_text(text):

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s!?;:"\'()-]', ' ', text) # keeps words and selected punctuation
    text = re.sub(r'([.,!?;:"\'()-])', r' \1 ', text)
    tokens = text.split()
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
    for author_file in author_files:
        path_file = os.path.join(path_dir, author_file)
        token_file = read_tokens_from_file(path_file)
        token_list.append(token_file)
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
    UNK_TOKEN = "<UNK>"
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

def load_data_task_2(path_file_in):
    with open(path_file_in, "r", encoding="utf-8", errors="ignore") as f:
        data_list = json.load(f)
    return data_list

def fine_tune_tokens(data_list):
    chunk_tokens_list = []
    for chunk in data_list['chunks']:
        chunk_tokens_list.append(clean_text(chunk))
    return chunk_tokens_list


def task_2(data_list, word_to_index, embeddings, path_file_out):
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

class CustomDataset(Dataset):
    def __init__(self, contexts_target_pairs):
        self.contexts_target_pairs = contexts_target_pairs
    
    def __len__(self):
        return len(self.contexts_target_pairs)
    
    def __getitem__(self, index):
        context, target = self.contexts_target_pairs[index]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def get_dataloader(contexts_target_pairs, batch_size=256, shuffle=True):
    print("Batching data...")
    dataset = CustomDataset(contexts_target_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class CBOW(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.context_embeddings = nn.Embedding(V, d)
        self.target_embeddings = nn.Embedding(V, d)
        with torch.no_grad():
            self.context_embeddings.weight.data.uniform_(-0.5/np.sqrt(d), 0.5/np.sqrt(d)) 
            self.target_embeddings.weight.zero_()

    def forward(self, target_word, context_words, negative_samples):
        context_embeddings = self.context_embeddings(context_words)
        target_embeddings = self.target_embeddings(target_word)
        negative_embeddings = self.target_embeddings(negative_samples)

        h = context_embeddings.mean(dim= 1)
        positive_scores = h.unsqueeze(1) @ target_embeddings.unsqueeze(2)
        positive_scores = positive_scores.squeeze()
        negative_scores = negative_embeddings @ h.unsqueeze(2)
        negative_scores = negative_scores.squeeze(2)

        return positive_scores, negative_scores

def sample_negatives(distribution, batch_size=256, K=10):
    negative_samples = torch.multinomial(distribution, num_samples=batch_size*K, replacement=True)
    return negative_samples.reshape(batch_size, K)

def compute_loss(positive_scores, negative_scores):
    positive_loss = F.logsigmoid(positive_scores)
    negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
    loss = -(positive_loss+negative_loss).mean()
    return loss

def train(epochs, model, train_dataloader, distribution, optimizer, K=10, timeout = 1500):
    best_train_loss = float('inf')
    print(f"timeout:{timeout}")

    start = time.time()

    for epoch in range(epochs):
        print(time.time() - start)
        if (time.time() - start > timeout):
            print("Training timed out...")
            break
        print(f"Training epoch:{epoch+1}")
        model.train()
        train_loss = 0.0
        for (context_words, target_word) in train_dataloader:
            negative_samples = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(target_word, context_words, negative_samples)
            loss = compute_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss/len(train_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}")

        if avg_train_loss <= best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "best_model_sim.pt")
        else:
            break



def fine_tune(epochs, model, train_dataloader, distribution, optimizer, K=10, timeout = 1500):
    best_train_loss = float('inf')
    print(f"timeout:{timeout}")

    start = time.time()

    for epoch in range(epochs):
        print(time.time() - start)
        if (time.time() - start > timeout):
            print("Training timed out...")
            break
        print(f"Training epoch:{epoch+1}")
        model.train()
        train_loss = 0.0
        for (context_words, target_word) in train_dataloader:
            negative_samples = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(target_word, context_words, negative_samples)
            loss = compute_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss/len(train_dataloader) 

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}")
    
        if avg_train_loss <= best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "ft_model_sim.pt")
        else:
            break
    

def save_embeddings(model, word_to_index, path = ".", file_name = "task_1.pt"):
    model.eval()
    embeddings = model.context_embeddings.weight
    file_path = os.path.join(path, file_name)
    torch.save({"embeddings" : embeddings, "word_to_index" : word_to_index}, file_path)

def load_embeddings(file_path = "./task_1.pt"):
    embed_dict = torch.load(file_path)
    return embed_dict["word_to_index"], embed_dict["embeddings"]

def cosine_similarity(word1, word2, word_to_index, embeddings):
    i1 = word_to_index[word1]
    e1 = embeddings[i1].unsqueeze(0)
    i2 = word_to_index[word2]
    e2 = embeddings[i2].unsqueeze(0)

    return F.cosine_similarity(e1, e2).item()

def train_val(epochs, model, train_dataloader, val_dataloader, distribution, optimizer, K=10, patience=3, timeout=1500):
    best_val_loss = float('inf')
    patience_counter = 0
    
    start = time.time()

    # Training phase
    for epoch in range(epochs):
        print(time.time() - start)
        if (time.time() - start > timeout):
            print("Training timed out...")
            break
        print(f"Training epoch:{epoch+1}")
        model.train()
        train_loss = 0.0
        for (context_words, target_word) in train_dataloader:
            negative_samples = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(target_word, context_words, negative_samples)
            loss = compute_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase    
        print(f"Validation epoch:{epoch+1}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (context_words, target_word) in val_dataloader:
                negative_sample = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
                positive_scores, negative_scores = model(target_word, context_words, negative_sample)
                loss = compute_loss(positive_scores, negative_scores)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_val.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}, Val Loss: {avg_val_loss:.2f}, patience Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def finetune_val(epochs, model, train_dataloader, val_dataloader, distribution, optimizer, K=10, patience=3, timeout=1500):
    best_val_loss = float('inf')
    patience_counter = 0
    
    start = time.time()

    # Training phase
    for epoch in range(epochs):
        print(time.time() - start)
        if (time.time() - start > timeout):
            print("Training timed out...")
            break
        print(f"Training epoch:{epoch+1}")
        model.train()
        train_loss = 0.0
        for (context_words, target_word) in train_dataloader:
            negative_samples = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(target_word, context_words, negative_samples)
            loss = compute_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase    
        print(f"Validation epoch:{epoch+1}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (context_words, target_word) in val_dataloader:
                negative_sample = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
                positive_scores, negative_scores = model(target_word, context_words, negative_sample)
                loss = compute_loss(positive_scores, negative_scores)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "ft_model_val.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}, Val Loss: {avg_val_loss:.2f}, patience Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break



def test_cosine_similarity(word_to_index, embeddings):
    print(cosine_similarity("king", "queen", word_to_index, embeddings))
    print(cosine_similarity("man", "woman", word_to_index, embeddings))
    e_king = embeddings[word_to_index["king"]].unsqueeze(0)
    e_queen = embeddings[word_to_index["queen"]].unsqueeze(0)
    e_man = embeddings[word_to_index["man"]].unsqueeze(0)
    e_woman = embeddings[word_to_index["woman"]].unsqueeze(0)
    e_test = e_king - e_man + e_woman
    print(F.cosine_similarity(e_test, e_queen).item())
    print(F.cosine_similarity(e_man, e_queen).item())
    print(F.cosine_similarity(e_king, e_woman).item())
    print(cosine_similarity("dog", "friend", word_to_index, embeddings))
    print(cosine_similarity("snake", "dog", word_to_index, embeddings))
    print(cosine_similarity("<UNK>", "dog", word_to_index, embeddings))

def build_finetune_counter(tokens_list):
    counter = Counter()
    for tokens in tokens_list:
        counter.update(tokens)
    return counter