import utils as u
import torch
import sys
from pathlib import Path
from config import *


raw_path_dataset = sys.argv[1]
abs_path_dataset = Path(raw_path_dataset).resolve()

split_ratio = SPLIT_RATIO
window = WINDOW_SIZE
K = K
d = DIMENSION
batch_size = BATCH_SIZE
epochs = EPOCHS
timeout = TIMEOUT
patience = PATIENCE

tokens_list = u.load_tokens(abs_path_dataset)
bigram_tokens_list = u.bigrams(tokens_list)

train_tokens_list, val_tokens_list = u.train_val_split(tokens_list, split_ratio=split_ratio)
word_to_index, index_to_word, counter = u.build_vocabulary_indices(train_tokens_list, min_freq=2)
train_token_indices = u.build_author_token_indices(train_tokens_list, word_to_index)
val_tokens_indices = u.build_author_token_indices(val_tokens_list, word_to_index)

train_contexts_target_pairs = u.build_contexts_target_pairs(train_token_indices, window=window, word_to_index=word_to_index)
val_contexts_target_pairs = u.build_contexts_target_pairs(val_tokens_indices, window=window, word_to_index=word_to_index)

batched_train_ct_pairs = u.get_dataloader(train_contexts_target_pairs, batch_size=batch_size, shuffle=True)
batched_val_ct_pairs = u.get_dataloader(val_contexts_target_pairs, batch_size=batch_size, shuffle=True)
negative_sampling_distribution = u.build_distribution(word_to_index, counter)

model = u.CBOW(V=len(word_to_index), d=d)
optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
u.train_val(epochs, model, batched_train_ct_pairs, batched_val_ct_pairs, negative_sampling_distribution, optimzer, word_to_index, K=K, patience=patience, timeout=timeout)
