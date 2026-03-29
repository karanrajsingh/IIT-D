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

device = None
word_to_idx = None
idx_to_word = None


def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"

def encode(text):
    global word_to_idx
    # tokens = text.lower().split()
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    unk_idx = word_to_idx["<unk>"]
    return [word_to_idx.get(token, unk_idx) for token in tokens]

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

def get_glove_embeddings(path, embedding_dim=100):
    global word_to_idx, idx_to_word
    word_to_idx = {}
    idx_to_word = []
    embeddings = []

    for token in ["<pad>", "<unk>", "<sos>", "<eos>"]:
        word_to_idx[token] = len(word_to_idx)
        idx_to_word.append(token)
    embeddings.append(torch.zeros(1, embedding_dim))
    embeddings.append(torch.randn(1, embedding_dim))
    embeddings.append(torch.randn(1, embedding_dim))
    embeddings.append(torch.randn(1, embedding_dim))

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0].lower()
            vector = torch.tensor([float(x) for x in tokens[1:]]).unsqueeze(0)
            word_to_idx[word] = len(word_to_idx)
            idx_to_word.append(word)
            embeddings.append(vector)

    embedding_matrix = torch.cat(embeddings, dim=0)

    return embedding_matrix

def build_filtered_glove(glove_path, texts, embedding_dim=100):
    global word_to_idx, idx_to_word

    vocab = set()
    for text in texts:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        vocab.update(tokens)

    word_to_idx = {"<pad>":0, "<unk>":1, "<sos>":2, "<eos>":3}
    idx_to_word = list(word_to_idx.keys())

    embeddings = [
        torch.zeros(1, embedding_dim),
        torch.randn(1, embedding_dim),
        torch.randn(1, embedding_dim),
        torch.randn(1, embedding_dim)
    ]

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0].lower()

            if word in vocab:
                vector = torch.tensor([float(x) for x in tokens[1:]]).unsqueeze(0)
                word_to_idx[word] = len(word_to_idx)
                idx_to_word.append(word)
                embeddings.append(vector)
    final_embed = torch.cat(embeddings, dim=0)
    
    wi = {}
    wi["word_to_idx"] = word_to_idx
    wi["index_to_word"] = idx_to_word
    wi["embeddings"] = final_embed

    os.makedirs("./tokenizer_data", exist_ok=True)
    file = os.path.join("./tokenizer_data", "tokenizer_dict.pt")

    tokenizer_dict = {
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
        "embeddings" : final_embed
    }

    torch.save(tokenizer_dict, file)

    return final_embed

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
    
# def collate_fn(input_tokens, tokenizer, ground_truth_tokens = None):
#     global device, word_to_idx
#     padded_input = pad_sequence(input_tokens, batch_first=True, padding_value=word_to_idx['<pad>']).to(device)
#     padded_ground_truth = pad_sequence(ground_truth_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
#     return padded_input, padded_ground_truth

def get_collate_fn(tokenizer):
    def collate_fn(batch):
        global device, word_to_idx
        sources, targets = zip(*batch)

        sources_padded = pad_sequence(
            sources, batch_first=True, padding_value=word_to_idx['<pad>']
        )

        targets_padded = pad_sequence(
            targets, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        return sources_padded, targets_padded

    return collate_fn

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.source = df["source"].astype(str).tolist()
        self.translated = df["translated"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = self.source[index]
        translated = self.translated[index]
        source_ids = encode(source)
        # translated_ids = self.tokenizer(translated)
        translated_ids = self.tokenizer.encode(translated)
        translated_ids = [self.sos_id] + translated_ids + [self.eos_id]
        return torch.tensor(source_ids), torch.tensor(translated_ids)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_output, embedding_dim, hidden_dim, pretrained_embedding, tokenizer, encoder_layers=2, decoder_layers=2, dropout=0.2):
        super().__init__()
        
        global word_to_idx

        self.vocab_size_output = vocab_size_output
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.tokenizer = tokenizer

        self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=word_to_idx["<pad>"]) # to freeze or not
        self.decoder_embedding = nn.Embedding(vocab_size_output, embedding_dim, padding_idx=tokenizer.pad_token_id)

        # BiLSTM encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # # Context down projection
        # self.hidden_proj = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        # self.cell_proj = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.context_ln = nn.LayerNorm(2*hidden_dim)
        # LSTM decoder
        self.decoder = nn.LSTM(embedding_dim+hidden_dim*2, hidden_dim*2, num_layers=decoder_layers, batch_first=True, dropout=dropout)
        self.decoder_ln = nn.LayerNorm(2*hidden_dim)
        self.unembedding = nn.Linear(hidden_dim*2, vocab_size_output)

    def forward(self, input_tokens, ground_truth_tokens=None, teacher_forcing_prob=0.6):
        
        global device, word_to_idx
        batch_size, sequence_len = input_tokens.size()
        pad_mask = (input_tokens != word_to_idx['<pad>'])
        lengths = pad_mask.sum(dim=1).cpu()
        embedded_input = self.encoder_embedding(input_tokens)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded_input, lengths, batch_first=True, enforce_sorted=False)

        o, (h, c) = self.encoder(packed_input)
        
        
        # Build context and initialization states for decoder
        h = torch.cat((h[-2], h[-1]), dim=1)
        c = torch.cat((c[-2], c[-1]), dim=1)
        # h = self.hidden_proj(h)
        # c = self.cell_proj(c)
        context = self.context_ln(h).unsqueeze(1)

        # Initialize decoder layers
        h = h.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_input = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device) # First input token is <SOS>
        outputs = []
        sentence_bool = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # target_len = ground_truth_tokens.size(1) if ground_truth_tokens is not None else sequence_len
        max_len = input_tokens.size(1) - 1
        target_len = ground_truth_tokens.size(1) -1 if ground_truth_tokens is not None else max_len
        for t in range(target_len):
            decoder_embedding = self.decoder_embedding(decoder_input)
            # Concatenate context to input embedding
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

def split_heads(x, num_heads=4, head_dim=128):
    B, T, D = x.size()
    x = x.view(B, T, num_heads, head_dim)
    return x.transpose(1, 2)

class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size_output, embedding_dim, hidden_dim, pretrained_embedding, tokenizer, encoder_layers=2, decoder_layers=2, dropout=0.2, num_heads=4):
        super().__init__()
        
        global word_to_idx

        self.vocab_size_output = vocab_size_output
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.tokenizer = tokenizer
        self.num_heads = num_heads

        self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=word_to_idx["<pad>"]) # to freeze or not
        self.decoder_embedding = nn.Embedding(vocab_size_output, embedding_dim, padding_idx=tokenizer.pad_token_id)

        # BiLSTM encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_dim)
        
        # LSTM decoder
        self.decoder = nn.LSTM(embedding_dim+hidden_dim*2, hidden_dim*2, num_layers=decoder_layers, batch_first=True, dropout=dropout)
        # Attention layer
        self.attn = nn.Linear(hidden_dim*2 + hidden_dim*2, hidden_dim*2)
        self.v = nn.Linear(hidden_dim*2, 1)
        self.context_proj = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.context_ln = nn.LayerNorm(hidden_dim*2)

        self.decoder_ln = nn.LayerNorm(hidden_dim*2)
        self.unembedding = nn.Linear(hidden_dim*2, vocab_size_output)

    def forward(self, input_tokens, ground_truth_tokens=None, teacher_forcing_prob=0.6):
        
        global device, word_to_idx
        batch_size, sequence_len = input_tokens.size()
        pad_mask = (input_tokens != word_to_idx['<pad>'])
        lengths = pad_mask.sum(dim=1).cpu()
        embedded_input = self.encoder_embedding(input_tokens)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded_input, lengths, batch_first=True, enforce_sorted=False)

        packed_o, (h, c) = self.encoder(packed_input) 

        encoder_outputs, _ = pad_packed_sequence(packed_o, batch_first=True)
        encoder_outputs = self.enc_ln(encoder_outputs)

        # Build context and initialization states for decoder
        h = torch.cat((h[-2], h[-1]), dim=1)
        c = torch.cat((c[-2], c[-1]), dim=1)

        # Initialize decoder layers
        h = h.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_input = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device) # First input token is <SOS>
        outputs = []
        sentence_bool = torch.zeros(batch_size, dtype=torch.bool, device=device)

        max_len = input_tokens.size(1) - 1
        target_len = ground_truth_tokens.size(1) -1 if ground_truth_tokens is not None else max_len
        for t in range(target_len):
            decoder_embedding = self.decoder_embedding(decoder_input)
            h_last = h[-1]
            h_repeat = h_last.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            aplhas = torch.tanh(self.attn(torch.cat((h_repeat, encoder_outputs), dim=2))) 
            attention_scores = self.v(aplhas).squeeze(-1)
            # Mask padding
            attention_scores = attention_scores.masked_fill(encoder_outputs == 0, -1e9)
            attention_weights = torch.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) 
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
    
def train_epoch(model, train_loader, val_loader, test_loader, optimizer, criterion, device):
    model.to(device)
    model.train()
    train_loss = 0
    for source, translated in train_loader:
        source = source.to(device)
        translated = translated.to(device)
        optimizer.zero_grad()
        out = model(source, translated)
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
            source = source.to(device)
            translated = translated.to(device)
            out = model(source, translated, teacher_forcing_prob=0.0)
            logits = out
            targets = translated[:, 1:1+out.size(1)]
            # loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)),targets.reshape(-1))
            val_loss += loss.item()*source.size(0)
            out_inf = model(source, None, teacher_forcing_prob=0.0)
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
            source = source.to(device)
            translated = translated.to(device)
            out = model(source, None, teacher_forcing_prob=0.0)
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

def train_model(model, train_loader, val_loader, test_loader, pad_id, num_epochs=50, save_path="./checkpoints/", lr=5e-4):
    global device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    history = {key: [] for key in ["train_loss","val_loss", "bleu", "chrf", "ter", "test_bleu", "test_chrf", "test_ter"]}
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss, val_loss, bleu, chrf, ter, test_bleu, test_chrf, test_ter = train_epoch(model, train_loader, val_loader, test_loader, optimizer, criterion, device)
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

    def get_ticks(epochs):
        step = max(1, len(epochs)//10)
        return epochs[::step]

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
    train_M_df = load_data("./data/Marathi_train.csv")
    val_M_df = load_data("./data/Marathi_dev.csv")
    test_M_df = load_data("./data/Marathi_public_test.csv")
    train_H_df = clean_col(train_H_df)
    val_H_df = clean_col(val_H_df)
    test_H_df = clean_col(test_H_df)
    train_M_df = clean_col(train_M_df)
    val_M_df = clean_col(val_M_df)
    test_M_df = clean_col(test_M_df)    
    
    savepath = "./checkpoints"
    os.makedirs(savepath, exist_ok=True)

    model_dir = os.path.join(savepath, "model")
    os.makedirs(model_dir, exist_ok=True)
    tokenizer_H = build_tokenizer(train_H_df, save_dir=os.path.join(model_dir, "tokenizer", "Hindi"))

    training_H_dataset = CustomDataset(train_H_df, tokenizer_H)
    valid_H_dataset = CustomDataset(val_H_df, tokenizer_H)
    test_H_dataset = CustomDataset(test_H_df, tokenizer_H)
    
    batch_size = 32
    hidden_dim = 128
    epochs = 30
    lr = 5e-4

    train_H_dataloader = DataLoader(dataset=training_H_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=get_collate_fn(tokenizer_H))
    valid_H_dataloader = DataLoader(dataset=valid_H_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(tokenizer_H))
    test_H_dataloader = DataLoader(dataset=test_H_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(tokenizer_H))

    embedding = build_filtered_glove("./glove.6B.100d.txt", train_H_df["source"].tolist())
    model = Seq2Seq(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, pretrained_embedding=embedding, tokenizer=tokenizer_H)
    global word_to_idx, idx_to_word
    history = train_model(model, train_H_dataloader, valid_H_dataloader, test_H_dataloader, pad_id=tokenizer_H.pad_token_id, num_epochs=epochs, save_path=os.path.join(model_dir, "trained"), lr=lr)
    plot_and_save_all(history, os.path.join(model_dir, "trained", "plots"))
    

if __name__ == "__main__":
    main()


class Seq2Seq2(nn.Module):
    def __init__(self, vocab_size_output, embedding_dim, hidden_dim, pretrained_embedding, tokenizer, input_dict, encoder_layers=2, decoder_layers=2, dropout=0.2):
        super().__init__()

        self.vocab_size_output = vocab_size_output
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.tokenizer = tokenizer
        self.input_dict = input_dict

        self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False, padding_idx=input_dict["<pad>"]) # to freeze or not
        self.decoder_embedding = nn.Embedding(vocab_size_output, embedding_dim, padding_idx=tokenizer.pad_token_id)

        # BiLSTM encoder
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=encoder_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.context_ln = nn.LayerNorm(2*hidden_dim)
        # LSTM decoder
        self.decoder = nn.LSTM(embedding_dim+hidden_dim*2, hidden_dim*2, num_layers=decoder_layers, batch_first=True, dropout=dropout)
        self.decoder_ln = nn.LayerNorm(2*hidden_dim)
        self.unembedding = nn.Linear(hidden_dim*2, vocab_size_output)

    def forward(self, input_tokens, ground_truth_tokens=None, teacher_forcing_prob=0.6):
        
        device = input_tokens.device
        batch_size, sequence_len = input_tokens.size()
        pad_mask = (input_tokens != self.input_dict['<pad>'])
        lengths = pad_mask.sum(dim=1).cpu()
        embedded_input = self.encoder_embedding(input_tokens)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded_input, lengths, batch_first=True, enforce_sorted=False)

        o, (h, c) = self.encoder(packed_input)
        
        # Build context and initialization states for decoder
        h = torch.cat((h[-2], h[-1]), dim=1)
        c = torch.cat((c[-2], c[-1]), dim=1)
        context = self.context_ln(h).unsqueeze(1)

        # Initialize decoder layers
        h = h.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        decoder_input = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device) # First input token is <SOS>
        outputs = []
        sentence_bool = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_len = input_tokens.size(1) - 1
        target_len = ground_truth_tokens.size(1) -1 if ground_truth_tokens is not None else max_len
        for t in range(target_len):
            decoder_embedding = self.decoder_embedding(decoder_input)
            # Concatenate context to input embedding
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


def get_collate_fn2(tokenizer, word_2_idx):
    def collate_fn2(batch):
        sources, targets = zip(*batch)
        sources_padded = pad_sequence(sources, batch_first=True, padding_value=word_2_idx['<pad>'])
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
        return sources_padded, targets_padded

def encode2(text, word_2_idx):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    unk_idx = word_2_idx["<unk>"]
    return [word_2_idx.get(token, unk_idx) for token in tokens]

class CustomDataset2(Dataset):
    def __init__(self, df, tokenizer, word_2_idx):
        self.source = df["source"].astype(str).tolist()
        self.translated = df["translated"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.word_2_idx =word_2_idx
        self.sos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = self.source[index]
        translated = self.translated[index]
        source_ids = encode2(source, self.word_2_idx)
        # translated_ids = self.tokenizer(translated)
        translated_ids = self.tokenizer.encode(translated)
        translated_ids = [self.sos_id] + translated_ids + [self.eos_id]
        return torch.tensor(source_ids), torch.tensor(translated_ids)