import re
import json

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import sacrebleu
import matplotlib.pyplot as plt
from torch.nn.functional import scaled_dot_product_attention
from transformers import BertModel, BertTokenizer

device = None


def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"


def decode_tokens(token_ids, tokenizer):
    tokens = []
    for idx in token_ids:
        if idx == tokenizer.eos_token_id:
            break
        if idx not in [tokenizer.pad_token_id, tokenizer.bos_token_id]:
            tokens.append(idx)
    text =  tokenizer.decode(tokens, skip_special_tokens=True).strip()
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    return text


def load_data(path = "./data"):
    return pd.read_csv(path)

def clean_text(text):
    clean_H = "translate to Hindi: "
    clean_M = "translate to Marathi: "
    cleaned_text = text.replace(clean_H, "")
    cleaned_text = cleaned_text.replace(clean_M, "")
    return cleaned_text.strip()

def clean_col(train_df):
    train_df["source"] = train_df["source"].apply(clean_text)
    return train_df

def build_tokenizer(train_df, vocab_size=5000, save_dir="./tokenizer"):
    os.makedirs(save_dir, exist_ok=True)
    global tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(train_df["translated"].astype(str).tolist(), trainer)
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", bos_token="<sos>", eos_token="<eos>")
    fast_tokenizer.save_pretrained(save_dir)
    return fast_tokenizer
    

def get_collate_fn(tokenizer, bert_tokenizer):
    def collate_fn(batch):
        global device
        sources, targets = zip(*batch)
        sources_padded = pad_sequence(sources, batch_first=True, padding_value=bert_tokenizer.pad_token_id)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
        return sources_padded, targets_padded
    return collate_fn

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, bert_tokenizer):
        self.source = df["source"].astype(str).tolist()
        self.translated = df["translated"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.sos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = self.source[index]
        translated = self.translated[index]
        source_ids = self.bert_tokenizer.encode(source, add_special_tokens=True)
        translated_ids = self.tokenizer.encode(translated)
        translated_ids = [self.sos_id] + translated_ids + [self.eos_id]
        return torch.tensor(source_ids), torch.tensor(translated_ids)

def split_heads(x, num_heads=4, head_dim=128):
    B, T, D = x.size()
    x = x.view(B, T, num_heads, head_dim)
    return x.transpose(1, 2)

class Seq2SeqBERTAttention(nn.Module):
    def __init__(self, vocab_size_output, embedding_dim, hidden_dim, tokenizer, encoder_layers=2, decoder_layers=2, dropout=0.2, num_heads=4):
        super().__init__()

        self.vocab_size_output = vocab_size_output
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.tokenizer = tokenizer
        self.num_heads = num_heads

        self.encoder = BertModel.from_pretrained("bert-base-cased")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.bert_hidden = self.encoder.config.hidden_size
        
        self.enc_proj = nn.Linear(self.bert_hidden, hidden_dim)
        self.enc_ln = nn.LayerNorm(hidden_dim)
        self.init_proj = nn.Linear(self.bert_hidden, hidden_dim)

        self.decoder_embedding = nn.Embedding(vocab_size_output, embedding_dim, padding_idx=tokenizer.pad_token_id)

        
        # LSTM decoder
        self.decoder = nn.LSTM(embedding_dim+hidden_dim, hidden_dim, num_layers=decoder_layers, batch_first=True, dropout=dropout)
        # Attention layer
        # self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_ln = nn.LayerNorm(hidden_dim)

        self.decoder_ln = nn.LayerNorm(hidden_dim)

        self.unembedding = nn.Linear(hidden_dim, vocab_size_output)

    def forward(self, input_tokens, attention_mask, ground_truth_tokens=None, teacher_forcing_prob=0.6):
        
        global device
        batch_size = input_tokens.size(0)
        
        encoder_outputs = self.encoder(input_ids=input_tokens, attention_mask=attention_mask)
        # pooled = encoder_outputs.last_hidden_state[:, 0]
        # pooled = self.init_proj(pooled)
        # encoder_hidden_states = self.enc_proj(encoder_outputs.last_hidden_state)
        
        masked = encoder_outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = self.init_proj(pooled)
        encoder_hidden_states = self.enc_proj(encoder_outputs.last_hidden_state)
        encoder_hidden_states = self.enc_ln(encoder_hidden_states)

        # Initialize decoder layers
        h = pooled.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        c = torch.zeros_like(h)
        decoder_input = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device) # First input token is <SOS>
        
        outputs = []
        sentence_bool = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # target_len = ground_truth_tokens.size(1) if ground_truth_tokens is not None else sequence_len
        max_len = input_tokens.size(1) - 1
        target_len = ground_truth_tokens.size(1) -1 if ground_truth_tokens is not None else max_len
        for t in range(target_len):
            decoder_embedding = self.decoder_embedding(decoder_input)
            # Attention mechanism
            # query = h[-1].unsqueeze(1)
            # Q = self.q_proj(query)
            # K = self.k_proj(encoder_hidden_states)
            # V = self.v_proj(encoder_hidden_states)
            # Q = split_heads(Q, num_heads=self.num_heads, head_dim=self.hidden_dim//self.num_heads)  
            # K = split_heads(K, num_heads=self.num_heads, head_dim=self.hidden_dim//self.num_heads) 
            # V = split_heads(V, num_heads=self.num_heads, head_dim=self.hidden_dim//self.num_heads)
            # attn_mask = ~attention_mask.unsqueeze(1).unsqueeze(2)
            # attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            # context = scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
            # # context = context.squeeze(1)
            # context = context.transpose(1, 2).contiguous()
            # context = context.view(batch_size, 1, -1)  # (B,1,2H)
            # context = self.out_proj(context)

            h_last = h[-1]  # (B,H)

            h_repeat = h_last.unsqueeze(1).repeat(
                1, encoder_hidden_states.size(1), 1
            )  # (B,T,H)

            energy = torch.tanh(
                self.attn(torch.cat((h_repeat, encoder_hidden_states), dim=2))
            )  # (B,T,H)

            attn_scores = self.v(energy).squeeze(-1)  # (B,T)

            # Mask padding
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)  # (B,T)

            context = torch.bmm(
                attn_weights.unsqueeze(1),
                encoder_hidden_states
            )  # (B,1,H)

            context = self.context_proj(context)
            context = self.context_ln(context)


            concat_input = torch.cat([decoder_embedding, context], dim=-1)
            o, (h, c) = self.decoder(concat_input, (h, c))
            o = self.decoder_ln(o)
            logits = self.unembedding(o.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            prediction = logits.argmax(dim=-1)
            if ground_truth_tokens is not None: # ground truth is None during inference
                spliced_ground_truth = ground_truth_tokens[:, t+1]
                # Per sample bernoulli sampling
                mask = torch.rand(batch_size, device=device) < teacher_forcing_prob
                next_input = torch.where(mask, spliced_ground_truth, prediction)
            else:
                next_input = prediction
            
            eos_mask = (next_input == self.tokenizer.eos_token_id)
            next_input = torch.where(sentence_bool, torch.full_like(next_input, self.tokenizer.pad_token_id, device=device), next_input)
            sentence_bool = sentence_bool | eos_mask
            decoder_input = next_input.unsqueeze(1)
            if ground_truth_tokens is None and sentence_bool.all():
                break

        return torch.cat(outputs, dim=1)
    
def train_epoch(model, train_loader, val_loader, test_loader, optimizer, criterion, device, pad_id=0):
    model.to(device)
    model.train()
    train_loss = 0
    for source, translated in train_loader:
        attention_mask = (source != pad_id).bool()
        attention_mask = attention_mask.to(device)
        source = source.to(device)
        translated = translated.to(device)
        optimizer.zero_grad()
        out = model(source, attention_mask, translated)
        # logits = out[:, :-1].contiguous()
        # targets = translated[:, 1:].contiguous()
        logits = out
        targets = translated[:, 1:1 + out.size(1)]
        # loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = criterion(logits.reshape(-1, logits.size(-1)),targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        train_loss += loss.item()*source.size(0)
    avg_train_loss = train_loss/len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    all_predictions = []
    all_references = []
    with torch.no_grad():        
        for source, translated in val_loader:
            attention_mask = (source != pad_id).bool()
            attention_mask = attention_mask.to(device)
            source = source.to(device)
            translated = translated.to(device)
            out = model(source, attention_mask, translated)
            logits = out
            targets = translated[:, 1:1+out.size(1)]
            # loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)),targets.reshape(-1))
            val_loss += loss.item()*source.size(0)
            out_inf = model(source, attention_mask, None, teacher_forcing_prob=0.0)
            prediction = out_inf.argmax(dim=-1)
            for pred_seq, tgt_seq in zip(prediction, translated):
                pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                tgt_text = decode_tokens(tgt_seq.tolist(), model.tokenizer)
                all_predictions.append(pred_text)
                all_references.append(tgt_text)
    avg_val_loss = val_loss/len(val_loader.dataset)
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references]).score
    chrf = sacrebleu.corpus_chrf(all_predictions, [all_references]).score
    ter = sacrebleu.corpus_ter(all_predictions, [all_references]).score

    all_predictions = []
    all_references = []
    with torch.no_grad():        
        for source, translated in test_loader:
            attention_mask = (source != pad_id).bool()
            attention_mask = attention_mask.to(device)
            source = source.to(device)
            translated = translated.to(device)
            out = model(source, attention_mask, None, teacher_forcing_prob=0.0)
            prediction = out.argmax(dim=-1)
            for pred_seq, tgt_seq in zip(prediction, translated):
                pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                tgt_text = decode_tokens(tgt_seq.tolist(), model.tokenizer)
                all_predictions.append(pred_text)
                all_references.append(tgt_text)
    test_bleu = sacrebleu.corpus_bleu(all_predictions, [all_references]).score
    test_chrf = sacrebleu.corpus_chrf(all_predictions, [all_references]).score
    test_ter = sacrebleu.corpus_ter(all_predictions, [all_references]).score

    return avg_train_loss, avg_val_loss, bleu, chrf, ter, test_bleu, test_chrf, test_ter

def train_model(model, train_loader, val_loader, test_loader, num_epochs=50, save_path="./checkpoints/", lr=5e-4, pad_id_1=0, pad_id_2=0):
    global device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_1)
    history = {key: [] for key in ["train_loss","val_loss", "bleu", "chrf", "ter", "test_bleu", "test_chrf", "test_ter"]}
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, val_loss, bleu, chrf, ter, test_bleu, test_chrf, test_ter = train_epoch(model, train_loader, val_loader, test_loader, optimizer, criterion, device, pad_id_2)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bleu"].append(bleu)
        history["chrf"].append(chrf)
        history["ter"].append(ter)
        history["test_bleu"].append(test_bleu)
        history["test_chrf"].append(test_chrf)
        history["test_ter"].append(test_ter)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu:.2f}, chrF: {chrf:.2f}, TER: {ter:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history
            }, os.path.join(save_path, "best_model.pt"))

            with open(os.path.join(save_path, "best_history.json"), "w") as f:
                json.dump({"bleu": bleu, "chrf": chrf, "ter": ter, "test_bleu": test_bleu, "test_chrf": test_chrf, "test_ter": test_ter}, f)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history
        }, os.path.join(save_path, "last_model.pt"))

        with open(os.path.join(save_path, "last_history.json"), "w") as f:
            json.dump({"bleu": bleu, "chrf": chrf, "ter": ter, "test_bleu": test_bleu, "test_chrf": test_chrf, "test_ter": test_ter}, f)

    return history


def plot_and_save_all(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # helper for clean integer ticks
    def get_ticks(epochs):
        step = max(1, len(epochs)//10)
        return epochs[::step]

    # ------------------ 1. Loss ------------------
    fig = plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.xticks(get_ticks(epochs))
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close(fig)

    # ------------------ 2. Validation Metrics ------------------
    fig = plt.figure()
    plt.plot(epochs, history["bleu"], label="BLEU")
    plt.plot(epochs, history["chrf"], label="chrF")
    plt.plot(epochs, history["ter"], label="TER")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.xticks(get_ticks(epochs))
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(save_dir, "val_metrics.png"))
    plt.close(fig)

    # ------------------ 3. Test Metrics ------------------
    if "test_bleu" in history:
        fig = plt.figure()
        plt.plot(epochs, history["test_bleu"], label="Test BLEU")
        plt.plot(epochs, history["test_chrf"], label="Test chrF")
        plt.plot(epochs, history["test_ter"], label="Test TER")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.title("Test Metrics")
        plt.xticks(get_ticks(epochs))
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(save_dir, "test_metrics.png"))
        plt.close(fig)

    # ------------------ 4. Individual Metric Curves (optional but useful) ------------------
    metrics = ["bleu", "chrf", "ter"]

    for metric in metrics:
        fig = plt.figure()
        plt.plot(epochs, history[metric], label=f"Val {metric.upper()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} Curve")
        plt.xticks(get_ticks(epochs))
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(save_dir, f"{metric}.png"))
        plt.close(fig)


def main():
    set_device()
    global device
    print(f"Device: {device}")
    train_H_df = load_data("./data/Hindi_train.csv")
    val_H_df = load_data("./data/Hindi_dev.csv")
    test_H_df = load_data("./data/Hindi_public_test.csv")
    train_H_df = clean_col(train_H_df)
    val_H_df = clean_col(val_H_df)
    test_H_df = clean_col(test_H_df)    
    
    savepath = "./checkpoints"
    os.makedirs(savepath, exist_ok=True)

    model_dir = os.path.join(savepath, "model_3")
    os.makedirs(model_dir, exist_ok=True)
    tokenizer_H = build_tokenizer(train_H_df, save_dir=os.path.join(model_dir, "tokenizer", "Hindi"))
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    training_H_dataset = CustomDataset(train_H_df, tokenizer_H, bert_tokenizer)
    valid_H_dataset = CustomDataset(val_H_df, tokenizer_H, bert_tokenizer)
    test_H_dataset = CustomDataset(test_H_df, tokenizer_H, bert_tokenizer)
    
    batch_size = 64
    hidden_dim = 256
    epochs = 100
    lr = 5e-4

    train_H_dataloader = DataLoader(dataset=training_H_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=get_collate_fn(tokenizer_H, bert_tokenizer))
    valid_H_dataloader = DataLoader(dataset=valid_H_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(tokenizer_H, bert_tokenizer))
    test_H_dataloader = DataLoader(dataset=test_H_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(tokenizer_H, bert_tokenizer))

    model = Seq2SeqBERTAttention(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, tokenizer=tokenizer_H, num_heads=1)
    history = train_model(model, train_H_dataloader, valid_H_dataloader, test_H_dataloader, pad_id_1=tokenizer_H.pad_token_id, pad_id_2=bert_tokenizer.pad_token_id, num_epochs=epochs, save_path=os.path.join(model_dir, "trained"), lr=lr)
    plot_and_save_all(history, os.path.join(model_dir, "trained", "plots"))
    

if __name__ == "__main__":
    main()
