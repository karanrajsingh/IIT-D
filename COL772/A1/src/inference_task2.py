import sys
from pathlib import Path
import utils as u
import torch
from config import *

window = WINDOW_SIZE
d = DIMENSION
batch_size = BATCH_SIZE
epochs = EPOCHS
timeout = TIMEOUT
ft_epochs = FT_EPOCHS
K = K

test_path = sys.argv[1]
abs_test_path = Path(test_path).resolve()
file_path_in = abs_test_path
dir_path = sys.argv[2]
abs_dir_path = Path(dir_path).resolve()
abs_dir_path.mkdir(parents=True, exist_ok=True)
file_path_out = abs_dir_path / "task2_predictions.json"
script_path = Path(__file__).resolve()
root = script_path.parent.parent
file_path_embed = root / "embedding_dict.pt"
word_to_index_loaded, embeddings_loaded = u.load_embeddings(file_path=file_path_embed)
data_list = u.load_data_task_2(file_path_in)


file_path_weights = root / "best_model_sim.pt"
weights = torch.load(file_path_weights, weights_only=True)
model = u.CBOW(V=weights['context_embeddings.weight'].shape[0], d=weights['context_embeddings.weight'].shape[1])
model.load_state_dict(weights)
chunk_token_list = u.fine_tune_tokens(data_list)
token_indices = u.build_author_token_indices(chunk_token_list, word_to_index_loaded)
contexts_target_pairs = u.build_contexts_target_pairs(token_indices, window=window, word_to_index=word_to_index_loaded)
batched_train_ct_pairs = u.get_dataloader(contexts_target_pairs, batch_size=batch_size, shuffle=True)
counter = u.build_finetune_counter(chunk_token_list)
negative_sampling_distribution = u.build_distribution(word_to_index_loaded, counter)
optimzer = torch.optim.Adam(model.parameters(), lr = 0.002)
u.fine_tune(ft_epochs, model, batched_train_ct_pairs, negative_sampling_distribution, optimzer, K=10, timeout=timeout)
u.save_embeddings(model, word_to_index_loaded, file_name="ft_embedding_dict.pt")
file_path_embed = root / "ft_embedding_dict.pt"
word_to_index_loaded, embeddings_loaded = u.load_embeddings(file_path=file_path_embed)
u.task_2(data_list, word_to_index_loaded, embeddings_loaded, file_path_out)