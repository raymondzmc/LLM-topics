import os
import torch
from tqdm import tqdm

KEYS = ['context', 'next_word', 'next_word_probs', 'input_embeddings']
CHUNK_SIZE = 2000


def save_processed_dataset(data_dict, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    for key in KEYS:
        values = data_dict.get(key, [])
        key_path = os.path.join(data_path, key)
        os.makedirs(key_path, exist_ok=True)
        for i in tqdm(list(range(0, len(values), CHUNK_SIZE))):
            torch.save(values[i:i+CHUNK_SIZE], os.path.join(key_path, f"{i}.pt"))
        print(f"Successfully saved {key} values")


def load_processed_dataset(data_path, layer_idx=None):
    data_dict = {}
    
    for key in KEYS:
        if key == 'input_embeddings' and layer_idx is not None:
            key_dir = os.path.join(data_path, key, f"{layer_idx}")
        else:
            key_dir = os.path.join(data_path, key)

        # Collect all chunk files (e.g., 0.pt, 1000.pt, 2000.pt, ...)
        chunk_files = [
            f for f in os.listdir(key_dir) 
            if f.endswith('.pt') and f.split('.')[0].isdigit()
        ]
        # Sort chunk files by the numeric portion in the filename
        chunk_files.sort(key=lambda x: int(x.split('.')[0]))
        
        values = []
        for chunk_file in tqdm(chunk_files):
            chunk_path = os.path.join(key_dir, chunk_file)
            chunk_data = torch.load(chunk_path, weights_only=False)
            values.extend(chunk_data)

        data_dict[key] = values
        print(f"Successfully loaded {key} values")

    assert (
        len(data_dict['context']) == len(data_dict['next_word']) == len(data_dict['next_word_probs']) == len(data_dict['input_embeddings'])
    ), "The lengths of the context, next_word, next_word_probs, and input_embeddings lists are not the same"

    return data_dict
