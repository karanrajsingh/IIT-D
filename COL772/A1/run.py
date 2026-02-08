# from utils import load_tokens, build_vocabulary_indices, build_author_token_indices
import utils as u
import train_model as tm
import time
import torch
import torch.nn.functional as F

path_dir = "./data/train_data/"
split_ratio = 0.8
window = 5
d = 150
batch_size = 256
epochs = 200

# From raw data to training and validation examples
tokens_list = u.load_tokens(path_dir)
bigram_tokens_list = u.bigrams(tokens_list)
trigrams_tokens_list = u.trigrams(tokens_list)

# tokens_list = bigram_tokens_list
# tokens_list = trigrams_tokens_list
# breakpoint()

train_tokens_list, val_tokens_list = u.train_val_split(tokens_list, split_ratio=split_ratio)
word_to_index, index_to_word, counter = u.build_vocabulary_indices(train_tokens_list, min_freq=2)
train_token_indices = u.build_author_token_indices(train_tokens_list, word_to_index)
val_tokens_list = u.build_author_token_indices(val_tokens_list, word_to_index)

# breakpoint()

# train_token_indices_dropout = u.apply_unk_dropout(train_token_indices, word_to_index)
train_token_indices_dropout = train_token_indices
# breakpoint()

train_contexts_target_pairs = u.build_contexts_target_pairs(train_token_indices_dropout, window=window, word_to_index=word_to_index)
val_contexts_target_pairs = u.build_contexts_target_pairs(val_tokens_list, window=window, word_to_index=word_to_index)
# breakpoint()


batched_train_ct_pairs = tm.get_dataloader(train_contexts_target_pairs, batch_size=batch_size, shuffle=True)
batched_val_ct_pairs = tm.get_dataloader(val_contexts_target_pairs, batch_size=batch_size, shuffle=True)
negative_sampling_distribution = u.build_distribution(word_to_index, counter)

# model = tm.CBOW(V=len(word_to_index), d=d)
# # optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
# # tm.train(epochs, model, batched_train_ct_pairs, batched_val_ct_pairs, negative_sampling_distribution, optimzer, K=10, patience=3)

# weights = torch.load("./best_model_256_lr2e3.pt", weights_only=True)
# model.load_state_dict(weights)
# tm.save_embeddings(model, word_to_index, file_name="best_model_256_lr2e3_task_1.pt")
# word_to_index_loaded, embeddings_loaded = tm.load_embeddings(file_path="./best_model_256_lr2e3_task_1.pt")
# print(tm.cosine_similarity("king", "queen", word_to_index_loaded, embeddings_loaded))
# print(tm.cosine_similarity("man", "woman", word_to_index_loaded, embeddings_loaded))
# e_king = embeddings_loaded[word_to_index["king"]].unsqueeze(0)
# e_queen = embeddings_loaded[word_to_index["queen"]].unsqueeze(0)
# e_man = embeddings_loaded[word_to_index["man"]].unsqueeze(0)
# e_woman = embeddings_loaded[word_to_index["woman"]].unsqueeze(0)
# e_test = e_king - e_man + e_woman
# print(F.cosine_similarity(e_test, e_queen).item())
# print(F.cosine_similarity(e_man, e_queen).item())
# print(F.cosine_similarity(e_king, e_woman).item())
# print(tm.cosine_similarity("dog", "friend", word_to_index_loaded, embeddings_loaded))
# print(tm.cosine_similarity("snake", "dog", word_to_index_loaded, embeddings_loaded))
# print(tm.cosine_similarity("<UNK>", "dog", word_to_index_loaded, embeddings_loaded))



path_file_in = "./sample_inputs/sample_task1.json"
path_file_out = "./sample_inputs/task1_predictions.jsonl"
weights = torch.load("./best_model_256_lr2e3.pt", weights_only=True)
model = tm.CBOW(V=weights['context_embeddings.weight'].shape[0], d=weights['context_embeddings.weight'].shape[1])
model.load_state_dict(weights)
tm.save_embeddings(model, word_to_index, file_name="best_model_256_lr2e3_task_1.pt")
word_to_index_loaded, embeddings_loaded = tm.load_embeddings(file_path="./best_model_256_lr2e3_task_1.pt")
out_list = u.task_1(path_file_in, word_to_index_loaded, embeddings_loaded, path_file_out)

# # fine-tuning:
weights = torch.load("./best_model_256_lr2e3.pt", weights_only=True)
model = tm.CBOW(V=weights['context_embeddings.weight'].shape[0], d=weights['context_embeddings.weight'].shape[1])
model.load_state_dict(weights)
path_file_in = "./sample_inputs/sample_task2.json"
data_list = u.load_data_task_2(path_file_in)
chunk_token_list = u.fine_tune_tokens(data_list)
tokens_list = chunk_token_list
train_tokens_list, val_tokens_list = u.train_val_split(tokens_list, split_ratio=split_ratio)
train_token_indices = u.build_author_token_indices(train_tokens_list, word_to_index)
val_tokens_list = u.build_author_token_indices(val_tokens_list, word_to_index)
train_contexts_target_pairs = u.build_contexts_target_pairs(train_token_indices, window=window, word_to_index=word_to_index)
val_contexts_target_pairs = u.build_contexts_target_pairs(val_tokens_list, window=window, word_to_index=word_to_index)
batched_train_ct_pairs = tm.get_dataloader(train_contexts_target_pairs, batch_size=batch_size, shuffle=True)
batched_val_ct_pairs = tm.get_dataloader(val_contexts_target_pairs, batch_size=batch_size, shuffle=True)
negative_sampling_distribution = u.build_distribution(word_to_index, counter)
optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
tm.train(epochs, model, batched_train_ct_pairs, batched_val_ct_pairs, negative_sampling_distribution, optimzer, K=10, patience=3)
path_file_out = "./sample_inputs/task2_predictions.json"
u.task_2(data_list, word_to_index_loaded, embeddings_loaded, path_file_out)

