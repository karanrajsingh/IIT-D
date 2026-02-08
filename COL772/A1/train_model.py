import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from datetime import datetime


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
        # Input context embeddings (W)
        self.context_embeddings = nn.Embedding(V, d)
        # Output target embeddings (W')
        self.target_embeddings = nn.Embedding(V, d)
        with torch.no_grad():
            self.context_embeddings.weight.data.uniform_(-0.5/np.sqrt(d), 0.5/np.sqrt(d)) 
            self.target_embeddings.weight.zero_()

    def forward(self, target_word, context_words, negative_samples):
        # Setup embeddings for a batch of input words
        context_embeddings = self.context_embeddings(context_words)
        target_embeddings = self.target_embeddings(target_word)
        negative_embeddings = self.target_embeddings(negative_samples)

        h = context_embeddings.mean(dim= 1)

        # Check dimensions
        positive_scores = h.unsqueeze(1) @ target_embeddings.unsqueeze(2)
        positive_scores = positive_scores.squeeze()
        negative_scores = negative_embeddings @ h.unsqueeze(2)
        negative_scores = negative_scores.squeeze(2)


        return positive_scores, negative_scores

def sample_negatives(distribution, batch_size=256, K=10):
    # print("Sampling negatives...")
    negative_samples = torch.multinomial(distribution, num_samples=batch_size*K, replacement=True)
    return negative_samples.reshape(batch_size, K)

def compute_loss(positive_scores, negative_scores):
    # print("Computing loss...")
    positive_loss = F.logsigmoid(positive_scores)
    negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
    loss = -(positive_loss+negative_loss).mean()
    return loss

def train(epochs, model, train_dataloader, val_dataloader, distribution, optimizer, K=10, patience=3):
    print("TRAINING STARTS:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training phase
    for epoch in range(epochs):
        print(f"Training epoch:{epoch+1}")
        model.train()
        train_loss = 0.0
        for batch_idx, (context_words, target_word) in enumerate(train_dataloader):
            # print(f"Train::epoch: {epoch+1}/{epochs}, batch: {batch_idx}/{len(train_dataloader)}")
            negative_samples = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
            optimizer.zero_grad()
            positive_scores, negative_scores = model(target_word, context_words, negative_samples)
            loss = compute_loss(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_dataloader)
        # avg_train_loss = train_loss 

        # Validation phase    
        print(f"Validation epoch:{epoch+1}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (context_words, target_word) in enumerate(val_dataloader):
                # print(f"Validation::epoch: {epoch+1}/{epochs}, batch: {batch_idx}/{len(val_dataloader)}")
                negative_sample = sample_negatives(distribution, batch_size=target_word.size(0), K=K)
                positive_scores, negative_scores = model(target_word, context_words, negative_sample)
                loss = compute_loss(positive_scores, negative_scores)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        # avg_val_loss = val_loss


        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}, Val Loss: {avg_val_loss:.2f}, patience Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("TRAINING ENDS:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))