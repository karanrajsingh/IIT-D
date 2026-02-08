# from utils import load_tokens, build_vocabulary_indices, build_author_token_indices
import utils as u
import train_model as tm
import time
import torch

path_dir = "./data/train_data/"
split_ratio = 0.8
window = 5
d = 150
batch_size = 4096
lr = 5e-4
epochs = 200

# From raw data to training and validation examples
tokens_list = u.load_tokens(path_dir)
train_tokens_list, val_tokens_list = u.train_val_split(tokens_list, split_ratio=split_ratio)
word_to_index, index_to_word, counter = u.build_vocabulary_indices(train_tokens_list, min_freq=2)
train_token_indices = u.build_author_token_indices(train_tokens_list, word_to_index)
val_tokens_list = u.build_author_token_indices(val_tokens_list, word_to_index)

# train_token_indices_dropout = u.apply_unk_dropout(train_token_indices, word_to_index)
train_token_indices_dropout = train_token_indices
# breakpoint()

train_contexts_target_pairs = u.build_contexts_target_pairs(train_token_indices_dropout, window=window, word_to_index=word_to_index)
val_contexts_target_pairs = u.build_contexts_target_pairs(val_tokens_list, window=window, word_to_index=word_to_index)
breakpoint()


batched_train_ct_pairs = tm.get_dataloader(train_contexts_target_pairs, batch_size=batch_size, shuffle=True)
batched_val_ct_pairs = tm.get_dataloader(val_contexts_target_pairs, batch_size=batch_size, shuffle=True)
negative_sampling_distribution = u.build_distribution(word_to_index, counter)

model = tm.CBOW(V=len(word_to_index), d=d)
optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
tm.train(epochs, model, batched_train_ct_pairs, batched_val_ct_pairs, negative_sampling_distribution, optimzer, K=10, patience=3)
