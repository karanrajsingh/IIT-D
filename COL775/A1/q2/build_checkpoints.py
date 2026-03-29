import torch
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
from training_4 import load_data, CustomDataset, get_collate_fn, decode_tokens
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer
import pandas as pd
from beam_search import generate_beam_predictions_bert2, generate_beam_predictions_lstm
import sacrebleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

c1 = torch.load("./tokenizer_data/tokenizer_dict.pt", map_location=device)

c2 = torch.load("./checkpoints/model_1/trained/last_model.pt", map_location=device)
c2["word_to_idx"] = c1["word_to_idx"]
c2["idx_to_word"] = c1["idx_to_word"]
c2["embeddings"]   = c1["embeddings"]

os.makedirs("./updated_checkpoints", exist_ok=True)
torch.save(c2, os.path.join("./updated_checkpoints", "updated_model_1.pt"))

c2 = torch.load("./checkpoints/model_2/trained/last_model.pt", map_location=device)
c2["word_to_idx"] = c1["word_to_idx"]
c2["idx_to_word"] = c1["idx_to_word"]
c2["embeddings"]   = c1["embeddings"]

os.makedirs("./updated_checkpoints", exist_ok=True)
torch.save(c2, os.path.join("./updated_checkpoints", "updated_model_2.pt"))