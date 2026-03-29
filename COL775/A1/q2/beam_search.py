import torch
from training_4 import Seq2SeqBERTAttention, load_data, CustomDataset, get_collate_fn, decode_tokens, clean_col
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer
import torch.nn.functional as F
import sacrebleu
import json
import os
import numpy as np
import time
import pandas as pd

device = ""

def set_device():
    global device
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"


def load_tokenizer(load_dir="./tokenizer"):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(load_dir)
    return tokenizer

def length_norm(score, length, alpha=0.7):
    return score/(np.pow(length, alpha))

def beam_search_decode_lstm(
    model,
    source,
    beam_size=5,
    max_token_len=100,
    alpha=0.7,
    pad_id = 0
):
    model.eval()
    global device

    sos_id = model.tokenizer.bos_token_id
    eos_id = model.tokenizer.eos_token_id

    with torch.no_grad():
        pad_mask = (source != pad_id)
        lengths = pad_mask.sum(dim=1).cpu()
        embedded = model.encoder_embedding(source)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_o, (h, c) = model.encoder(packed)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_o, batch_first=True)
        encoder_outputs = model.enc_ln(encoder_outputs)

        h_cat = torch.cat((h[-2], h[-1]), dim=1)
        c_cat = torch.cat((c[-2], c[-1]), dim=1)
        h = h_cat.unsqueeze(0).repeat(model.decoder_layers, 1, 1)
        c = c_cat.unsqueeze(0).repeat(model.decoder_layers, 1, 1)

    beams = [([sos_id], h, c, 0.0)]
    completed = []
    for t in range(max_token_len):
        new_beams = []
        for sequence, h, c, score in beams:
            if sequence[-1] == eos_id:
                completed.append((sequence, score))
                continue
            last_token = torch.tensor([[sequence[-1]]], device=device)
            decoder_embedding = model.decoder_embedding(last_token)
            h_last = h[-1]
            h_repeat = h_last.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            energy = torch.tanh(model.attn(torch.cat((h_repeat, encoder_outputs), dim=2)))

            attn_scores = model.v(energy).squeeze(-1)

            # mask padding
            attn_scores = attn_scores.masked_fill(~pad_mask, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)

            context = torch.bmm(
                attn_weights.unsqueeze(1),
                encoder_outputs
            )

            context = model.context_proj(context)
            context = model.context_ln(context)

            # ---------- Decoder ----------
            concat_input = torch.cat([decoder_embedding, context], dim=-1)

            o, (h_new, c_new) = model.decoder(concat_input, (h, c))
            o = model.decoder_ln(o)

            logits = model.unembedding(o.squeeze(1))
            log_probs = F.log_softmax(logits, dim=-1)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            for i in range(beam_size):
                token_id = topk_ids[0, i].item()
                token_score = topk_log_probs[0, i].item()

                new_seq = sequence + [token_id]
                new_score = score + token_score

                new_beams.append((
                    new_seq,
                    h_new.clone(),
                    c_new.clone(),
                    new_score
                ))

        if not new_beams:
            break

        # prune beams
        beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_size]

        # early stopping
        if all(seq[-1] == eos_id for seq, _, _, _ in beams):
            completed.extend([(seq, score) for seq, _, _, score in beams])
            break

    # add remaining beams
    for seq, _, _, score in beams:
        completed.append((seq, score))

    # ---------- length normalization ----------
    completed = sorted(
        completed,
        key=lambda x: length_norm(x[1], len(x[0]), alpha),
        reverse=True
    )

    return completed[0][0]

def generate_beam_predictions_lstm(model, dataloader, beam_size, input_decoder, input_tokenizer):
    model.eval()
    global device
    predictions = []
    sources = []
    with torch.no_grad():
        for source, translated in dataloader:
            attention_mask = (source != 0).long().to(device)
            source = source.to(device)
            for i in range(source.size(0)):
                prediction_ids = beam_search_decode_lstm(model, source[i].unsqueeze(0), attention_mask[i].unsqueeze(0), beam_size=beam_size)
                prediction = decode_tokens(prediction_ids, model.tokenizer)
                filtered_ids = [tid for tid in source[i].cpu().tolist() if tid != input_tokenizer["<UNK>"] and tid != input_tokenizer["<PAD>"] and tid != input_tokenizer["<SOS>"] and tid != input_tokenizer["<EOS>"]]
                source_text = [input_decoder.get(idx, '<UNK>') for idx in filtered_ids]
                predictions.append(prediction)
                sources.append(source_text)
    return sources, predictions


def beam_search_decode_bert(model, source, attention_mask, beam_size, max_token_len=100):
    model.eval()
    global device
    sos_id = model.tokenizer.bos_token_id
    eos_id = model.tokenizer.eos_token_id
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=source, attention_mask=attention_mask)
        masked = encoder_outputs.last_hidden_state*attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = model.init_proj(pooled)
        encoder_hidden_states = model.enc_proj(encoder_outputs.last_hidden_state)
        encoder_hidden_states = model.enc_ln(encoder_hidden_states)
        h = pooled.unsqueeze(0).repeat(model.decoder_layers, 1, 1)
        c = torch.zeros_like(h)

    beams = [([sos_id], h, c, 0.0)]
    completed = []

    for t in range(max_token_len):
        new_beams = []
        for sequence, h, c, score in beams:
            if sequence[-1] == eos_id:
                completed.append((sequence, score))
                continue
            input_token = torch.tensor([[sequence[-1]]], device=device)
            decoder_embedding = model.decoder_embedding(input_token)
            h_last = h[-1]
            h_repeat = h_last.unsqueeze(1).repeat(1, encoder_hidden_states.size(1), 1)
            energy = torch.tanh(model.attn(torch.cat((h_repeat, encoder_hidden_states), dim=2)))
            attn_scores = model.v(energy).squeeze(-1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_hidden_states)
            context = model.context_proj(context)
            context = model.context_ln(context)
            concat_input = torch.cat([decoder_embedding, context], dim=-1)
            o, (h2, c2) = model.decoder(concat_input, (h, c)) # check copy
            o = model.decoder_ln(o)
            logits = model.unembedding(o.squeeze(1))
            log_probrobability = F.log_softmax(logits, dim=-1)
            topk_log_probability, topk_ids = torch.topk(log_probrobability, beam_size, dim=-1)
            for i in range(beam_size):
                token_id = topk_ids[0, i].item()
                token_score = topk_log_probability[0, i].item()
                new_sequence = sequence + [token_id]
                new_score = score + token_score
                new_beams.append((new_sequence, h2.clone(), c2.clone(), new_score)) # check copy
        beams = sorted(new_beams, key=lambda x: x[3], reverse=True)[:beam_size]
        if len(beams) == 0:
            break
        if all(seq[-1] == eos_id for seq, h, c, score in beams):
            break
    for sequence, h, c, score in beams:
        completed.extend([(sequence, score)])

    completed = sorted(completed, key=lambda x: length_norm(x[1], len(x[0])), reverse=True)
    return completed[0][0]


def generate_beam_predictions_bert(model, dataloader, beam_size, bert_tokenizer):
    model.eval()
    global device
    predictions = []
    translations = []
    sources = []
    with torch.no_grad():
        for source, translated in dataloader:
            attention_mask = (source != 0).long().to(device)
            source = source.to(device)
            for i in range(source.size(0)):
                src_text = bert_tokenizer.decode(source[i].detach().cpu().tolist(), skip_special_tokens=True)
                prediction_ids = beam_search_decode_bert(model, source[i].unsqueeze(0), attention_mask[i].unsqueeze(0), beam_size=beam_size)
                prediction = decode_tokens(prediction_ids, model.tokenizer)
                translation = decode_tokens(translated[i].tolist(), model.tokenizer)
                predictions.append(prediction)
                translations.append(translation)
                sources.append(src_text)
    return sources, predictions, translations

def generate_beam_predictions_bert2(model, dataloader, beam_size, bert_tokenizer, device):
    model.eval()
    predictions = []
    sources = []
    with torch.no_grad():
        for source, translated in dataloader:
            attention_mask = (source != bert_tokenizer.pad_token_id).bool().to(device)
            source = source.to(device)
            for i in range(source.size(0)):
                
                prediction_ids = beam_search_decode_bert(model, source[i].unsqueeze(0), attention_mask[i].unsqueeze(0), beam_size=beam_size)
                prediction = decode_tokens(prediction_ids, model.tokenizer)
                source = bert_tokenizer.decode(source[i].cpu().tolist(), skip_special_tokens=True)
                predictions.append(prediction)
                sources.append(source)
    return sources, predictions


def main():
    set_device()
    global device
    path_tokenizer = "./checkpoints/model_4/tokenizer/Hindi"
    output_tokenizer = load_tokenizer(load_dir=path_tokenizer)
    input_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    test_df = load_data("./data/Hindi_public_test.csv")
    test_df = clean_col(test_df)
    test_dataset = CustomDataset(test_df, output_tokenizer, input_tokenizer)
    batch_size = 64
    hidden_dim = 256
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(output_tokenizer, input_tokenizer))
    model = Seq2SeqBERTAttention(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, tokenizer=output_tokenizer, num_heads=1)
    model.to(device)
    path = "./checkpoints/model_4/trained/last_model.pt"
    checkpoint = torch.load(path, map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scores = {key : {} for key in [1, 5, 10, 20]}
    for beam_size in [1, 5, 10, 20]:
        t1 = time.time()
        print(f"Evaluating for beam size {beam_size}...")
        sources, predictions, translations = generate_beam_predictions_bert(model, test_dataloader, beam_size, input_tokenizer)
        bleu = sacrebleu.corpus_bleu(predictions, [translations]).score
        chrf = sacrebleu.corpus_chrf(predictions, [translations]).score
        ter = sacrebleu.corpus_ter(predictions, [translations]).score
        scores[beam_size]["bleu"] = bleu
        scores[beam_size]["chrf"] = chrf
        scores[beam_size]["ter"] = ter
        t2 = time.time()
        print(f"Scores for beam size: {beam_size} are bleu: {bleu}, chrf: {chrf}, ter: {ter}")
        print(f"Time taken for beam size: {beam_size} is {t2-t1} seconds.")
        df = pd.DataFrame()
        df["source"] = sources
        df["predictions"] = predictions
        df["translated"] = predictions
        os.makedirs("./beam_search", exist_ok=True)
        path = os.path.join("./beam_search", f"prediction_b{beam_size}.csv")
        df.to_csv(path)
    os.makedirs("./checkpoints", exist_ok=True)
    path = os.path.join("./checkpoints", "beam_results.json")
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()