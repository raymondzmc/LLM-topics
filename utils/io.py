import os
import torch
from tqdm import tqdm

KEYS = ['context', 'next_word', 'next_word_probs', 'input_embeddings']


def save_processed_dataset(data_dict, data_path, chunk_size=2000):
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    for key in KEYS:
        values = data_dict.get(key, [])
        key_path = os.path.join(data_path, key)
        os.makedirs(key_path, exist_ok=True)
        for i in tqdm(list(range(0, len(values), chunk_size))):
            torch.save(values[i:i+chunk_size], os.path.join(key_path, f"{i}.pt"))
        print(f"Successfully saved {key} values")


def load_processed_dataset(data_path, layer_idx=None, chunk_idx=None, verbose=True):
    data_dict = {}
    
    for key in KEYS:
        key_dir = os.path.join(data_path, key)
        # Collect all chunk files (e.g., 0.pt, 1000.pt, 2000.pt, ...)
        chunk_files = [
            f for f in os.listdir(key_dir) 
            if f.endswith('.pt') and f.split('.')[0].isdigit()
        ]
        # Sort chunk files by the numeric portion in the filename
        chunk_files.sort(key=lambda x: int(x.split('.')[0]))

        if chunk_idx is not None:
            chunk_files = [chunk_files[chunk_idx]]

        values = []
        for chunk_file in tqdm(chunk_files):
            chunk_path = os.path.join(key_dir, chunk_file)
            chunk_data = torch.load(chunk_path, weights_only=False)
            # if key == 'input_embeddings':
            #     import pdb; pdb.set_trace()

            # if key == 'input_embeddings' and layer_idx is not None and len(chunk_data[0].shape) > 2:
            #     chunk_data = [embedding[layer_idx] for embedding in chunk_data]

            values.extend(chunk_data)

        data_dict[key] = values
        if verbose:
            print(f"Successfully loaded {key} values")

    return data_dict