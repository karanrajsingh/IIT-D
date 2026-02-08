# from utils import load_tokens, build_vocabulary_indices, build_author_token_indices
import utils as u
import train_model as tm
import time
import torch
import torch.nn.functional as F

path_dir = "./data/train_data/"
split_ratio = 1.0
window = 5
K = 10
d = 150
batch_size = 256
epochs = 5

# From raw data to training and validation examples
tokens_list = u.load_tokens(path_dir)
bigram_tokens_list = u.bigrams(tokens_list)
trigrams_tokens_list = u.trigrams(tokens_list)

word_to_index, index_to_word, counter = u.build_vocabulary_indices(tokens_list, min_freq=2)
token_indices = u.build_author_token_indices(tokens_list, word_to_index)
contexts_target_pairs = u.build_contexts_target_pairs(token_indices, window=window, word_to_index=word_to_index)

batched_ct_pairs = tm.get_dataloader(contexts_target_pairs, batch_size=batch_size, shuffle=True)
negative_sampling_distribution = u.build_distribution(word_to_index, counter)

model = tm.CBOW(V=len(word_to_index), d=d)
optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
tm.train_simple(epochs, model, batched_ct_pairs, negative_sampling_distribution, optimzer, K=K, patience=3)

weights = torch.load("./best_model.pt", weights_only=True)
model.load_state_dict(weights)
tm.save_embeddings(model, word_to_index, file_name="best_model_task_1.pt")
word_to_index_loaded, embeddings_loaded = tm.load_embeddings(file_path="./best_model_task_1.pt")
print(tm.cosine_similarity("king", "queen", word_to_index_loaded, embeddings_loaded))
print(tm.cosine_similarity("man", "woman", word_to_index_loaded, embeddings_loaded))
e_king = embeddings_loaded[word_to_index["king"]].unsqueeze(0)
e_queen = embeddings_loaded[word_to_index["queen"]].unsqueeze(0)
e_man = embeddings_loaded[word_to_index["man"]].unsqueeze(0)
e_woman = embeddings_loaded[word_to_index["woman"]].unsqueeze(0)
e_test = e_king - e_man + e_woman
print(F.cosine_similarity(e_test, e_queen).item())
print(F.cosine_similarity(e_man, e_queen).item())
print(F.cosine_similarity(e_king, e_woman).item())
print(tm.cosine_similarity("dog", "friend", word_to_index_loaded, embeddings_loaded))
print(tm.cosine_similarity("snake", "dog", word_to_index_loaded, embeddings_loaded))
print(tm.cosine_similarity("<UNK>", "dog", word_to_index_loaded, embeddings_loaded))


# fine-tuning:
# Load model weight
# with new train and val loaders can train on the model
# check with existing weights if it improves


