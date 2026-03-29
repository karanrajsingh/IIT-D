import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import os
from torch.utils.data import DataLoader
from training_4 import Seq2SeqBERTAttention2 as S4
from training_3 import Seq2SeqBERTAttention2 as S3
from training_2 import Seq2SeqAttention2
from training import Seq2Seq2, clean_col
from transformers import PreTrainedTokenizerFast
from training_4 import load_data, get_collate_fn, decode_tokens
from training_4 import CustomDataset as CS4
from training_3 import CustomDataset as CS3
from training_2 import CustomDataset2 as CS2, get_collate_fn2
from training import CustomDataset2 as CS
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer
import pandas as pd
from beam_search import generate_beam_predictions_bert2, generate_beam_predictions_lstm
import sacrebleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(mode, checkpoint_path, df, decoding_strategy, output):
    if mode == 1:
        batch_size = 32
        hidden_dim = 128
        # save glove embedding + tokenizer and load here
        checkpoint = torch.load(checkpoint_path, map_location=device)
        input_tokenizer = checkpoint["word_to_idx"]
        output_tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer/Hindi")
        input_decoder = checkpoint["idx_to_word"]
        embedding = checkpoint["embeddings"]
        model = Seq2Seq2(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, pretrained_embedding=embedding, tokenizer=output_tokenizer, input_dict=input_tokenizer)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = CS(df, output_tokenizer, input_tokenizer)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn2(output_tokenizer, input_tokenizer))  
        with torch.no_grad():        
            source_list = []
            pred_text_list = []
            for source, _ in dataloader:
                source = source.to(device)
                out = model(source, None, teacher_forcing_prob=0.0)
                prediction = out.argmax(dim=-1)
                for src, pred_seq in zip(source, prediction):
                    pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                    filtered_ids = [tid for tid in src.cpu().tolist() if tid != input_tokenizer["<UNK>"] and tid != input_tokenizer["<PAD>"] and tid != input_tokenizer["<SOS>"] and tid != input_tokenizer["<EOS>"]]
                    source_text = [input_decoder.get(idx, '<UNK>') for idx in filtered_ids]
                    source_list.append(source_text)
                    pred_text_list.append(pred_text)
        df = pd.DataFrame()
        df["source"] = source_list
        df["translated"] = pred_text_list
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, "prediction_1.csv")
        df.to_csv(path)
        
    elif mode == 2:
        batch_size = 32
        hidden_dim = 128
        # save glove embedding + tokenizer and load here
        checkpoint = torch.load(checkpoint_path, map_location=device)
        input_tokenizer = checkpoint["word_to_idx"]
        output_tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer/Hindi")
        input_decoder = checkpoint["idx_to_word"]
        embedding = checkpoint["embeddings"]
        model = Seq2SeqAttention2(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, pretrained_embedding=embedding, tokenizer=output_tokenizer, input_dict=input_tokenizer)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = CS2(df, output_tokenizer, input_tokenizer)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn2(output_tokenizer, input_tokenizer))  
        with torch.no_grad():        
            source_list = []
            pred_text_list = []
            for source, _ in dataloader:
                source = source.to(device)
                out = model(source, None, teacher_forcing_prob=0.0)
                prediction = out.argmax(dim=-1)
                for src, pred_seq in zip(source, prediction):
                    pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                    filtered_ids = [tid for tid in src.cpu().tolist() if tid != input_tokenizer["<UNK>"] and tid != input_tokenizer["<PAD>"] and tid != input_tokenizer["<SOS>"] and tid != input_tokenizer["<EOS>"]]
                    source_text = [input_decoder.get(idx, '<UNK>') for idx in filtered_ids]
                    source_list.append(source_text)
                    pred_text_list.append(pred_text)
        df = pd.DataFrame()
        df["source"] = source_list
        df["translated"] = pred_text_list
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, "prediction_2.csv")
        df.to_csv(path)

    elif mode == 3:
        batch_size = 32
        hidden_dim = 256
        checkpoint = torch.load(checkpoint_path, map_location=device)
        output_tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer/Hindi")
        input_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = S3(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, tokenizer=output_tokenizer)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = CS3(df, output_tokenizer, input_tokenizer)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(output_tokenizer, input_tokenizer))  
        with torch.no_grad():        
            source_list = []
            pred_text_list = []
            for source, _ in dataloader:
                source = source.to(device)
                attention_mask = (source != input_tokenizer.pad_token_id).long()
                attention_mask = attention_mask.to(device)
                out = model(source, attention_mask, None, teacher_forcing_prob=0.0)
                prediction = out.argmax(dim=-1)
                for src, pred_seq in zip(source, prediction):
                    pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                    source_text = input_tokenizer.decode(src.cpu().tolist(), skip_special_tokens=True)  
                    source_list.append(source_text)
                    pred_text_list.append(pred_text)
        df = pd.DataFrame()
        df["source"] = source_list
        df["translated"] = pred_text_list
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, "prediction_3.csv")
        df.to_csv(path)
    elif mode == 4:
        batch_size = 32
        hidden_dim = 256
        checkpoint = torch.load(checkpoint_path, map_location=device)
        output_tokenizer = PreTrainedTokenizerFast.from_pretrained("./checkpoints/model_4/tokenizer/Hindi")
        input_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = S4(vocab_size_output=5000, embedding_dim=100, hidden_dim=hidden_dim, tokenizer=output_tokenizer)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = CS4(df, output_tokenizer, input_tokenizer)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=get_collate_fn(output_tokenizer, input_tokenizer))  
        with torch.no_grad():        
            source_list = []
            pred_text_list = []
            tgt_list = []
            for source, translated in dataloader:
                source = source.to(device)
                attention_mask = (source != input_tokenizer.pad_token_id).long()
                attention_mask = attention_mask.to(device)
                out = model(source, attention_mask, None, teacher_forcing_prob=0.0)
                prediction = out.argmax(dim=-1)
                for src, tgt_seq, pred_seq in zip(source, translated, prediction):
                    pred_text = decode_tokens(pred_seq.tolist(), model.tokenizer)
                    tgt_text = decode_tokens(tgt_seq.tolist(), model.tokenizer)
                    tgt_list.append(tgt_text)
                    source_text = input_tokenizer.decode(src.cpu().tolist(), skip_special_tokens=True)  
                    source_list.append(source_text)
                    pred_text_list.append(pred_text)
            bleu = sacrebleu.corpus_bleu(pred_text_list, [tgt_list]).score
            chrf = sacrebleu.corpus_chrf(pred_text_list, [tgt_list]).score
            ter = sacrebleu.corpus_ter(pred_text_list, [tgt_list]).score
            print(f"bleu: {bleu}, chrf: {chrf}, ter: {ter}")
        df = pd.DataFrame()
        df["source"] = source_list
        df["translated"] = pred_text_list
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, "prediction_4_1.csv")
        df.to_csv(path)
        # else:
        #     sources, predictions = generate_beam_predictions_bert2(model, dataloader, beam_size=5, bert_tokenizer=input_tokenizer, device=device)
        #     df = pd.DataFrame()
        #     df["source"] = sources
        #     df["translated"] = predictions
        #     os.makedirs(output, exist_ok=True)
        #     path = os.path.join(output, "prediction_4.csv")
        #     df.to_csv(path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=int, required=True)
    parser.add_argument("--decoding_strategy", type=str, choices=["greedy", "beam"], required=True)
    args = parser.parse_args()
    df = load_data(args.input)
    df = clean_col(df)

    model = load_model(args.mode, args.checkpoint, df, args.decoding_strategy, args.output)
