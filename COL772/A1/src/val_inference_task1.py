import sys
from pathlib import Path
import utils as u

test_path = sys.argv[1]
abs_test_path = Path(test_path).resolve()
file_path_in = abs_test_path

dir_path = sys.argv[2]
abs_dir_path = Path(dir_path).resolve()
abs_dir_path.mkdir(parents=True, exist_ok=True)
file_path_out = abs_dir_path / "val_task1_predictions.jsonl"

script_path = Path(__file__).resolve()
root = script_path.parent.parent
file_path_embed = root / "embedding_val_dict.pt"
word_to_index_loaded, embeddings_loaded = u.load_embeddings(file_path=file_path_embed)
u.task_1(file_path_in, word_to_index_loaded, embeddings_loaded, file_path_out)

# u.test_cosine_similarity(word_to_index_loaded, embeddings_loaded)
