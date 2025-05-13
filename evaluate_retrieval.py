import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
# scipy.special.kl_div and numpy.ma are no longer needed with batched KL.


# The old compute_kl_divergence function is removed as we use a batched version.

def compute_pairwise_kl_divergence_torch(P, device):
    """
    Compute pairwise KL divergence between rows of a matrix P using PyTorch.
    KL(P_i || P_j) = sum_k P_i[k] * (log P_i[k] - log P_j[k])
    This version computes the matrix row-by-row to be more memory-efficient.
    Args:
        P (torch.Tensor): Input tensor of shape (n_docs, n_features) where rows are distributions.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
    Returns:
        torch.Tensor: Pairwise KL divergence matrix of shape (n_docs, n_docs).
    """
    P = P.to(device)
    n_docs, n_features = P.shape
    P_norm = P / P.sum(dim=1, keepdims=True)
    P_stable = P_norm + 1e-12
    logP_all = P_stable.log()
    kl_matrix = torch.zeros((n_docs, n_docs), device=device)

    print(f"  Calculating KL divergence row-by-row for {n_docs} documents...")
    for i in tqdm(range(n_docs), desc="  KL Div Row", unit="doc", leave=False, dynamic_ncols=True):
        log_diff = logP_all[i, :].unsqueeze(0) - logP_all
        product = P_norm[i, :].unsqueeze(0) * log_diff
        kl_matrix[i, :] = product.sum(dim=-1)
    kl_matrix.fill_diagonal_(0)
    
    return kl_matrix


def compute_precision_at_k(retrieved_indices, query_labels, all_labels, k_values=[1, 5, 10]):
    """Compute precision@k for the retrieved documents."""
    results = {}
    query_label = query_labels
    
    for k in k_values:
        if k > len(retrieved_indices):
            continue
        top_k_indices = retrieved_indices[:k]
        top_k_labels = all_labels[top_k_indices]
        precision = np.mean(top_k_labels == query_label)
        results[f'precision@{k}'] = precision
        
    return results


def main(args):
    labels = np.loadtxt(args.label_file)
    similarity_matrix: np.ndarray | None = None # Initialize similarity matrix
    
    # Determine device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(args.bow_corpus_path), 'r', encoding='utf-8') as f:
        bow_corpus = [line.strip().split() for line in f]
        ignore_indices = [i for i, doc in enumerate(bow_corpus) if len(doc) == 0 or doc == ['null']]
        labels = np.array([int(label) for i, label in enumerate(labels) if i not in ignore_indices])

    if args.method == 'topic_distribution':
        output = torch.load(args.data_file, weights_only=False)
        if 'topic-document-matrix' not in output.keys():
            raise ValueError(f"topic-document-matrix not found in output keys at {args.data_file}")
        topic_distribution_np: np.ndarray = output['topic-document-matrix'].transpose()
        
        # Ensure topic_distribution_np aligns with filtered labels
        if topic_distribution_np.shape[0] != len(labels):
            if topic_distribution_np.shape[0] > len(labels) and len(ignore_indices) > 0:
                # Assuming topic_distribution_np was from an unfiltered corpus and ignore_indices applies to it.
                n_original_docs = topic_distribution_np.shape[0]
                valid_rows_mask = np.ones(n_original_docs, dtype=bool)
                if max(ignore_indices) < n_original_docs: # Check if ignore_indices are valid for this array
                    valid_rows_mask[ignore_indices] = False
                    topic_distribution_np = topic_distribution_np[valid_rows_mask, :]
                else:
                    raise ValueError(
                        f"ignore_indices contains indices out of bounds for topic_distribution_np. "
                        f"Max index: {max(ignore_indices)}, Original docs: {n_original_docs}"
                    )
            else:
                # This case implies a mismatch that cannot be resolved by ignore_indices as applied.
                raise ValueError(
                    f"Shape mismatch between topic distributions ({topic_distribution_np.shape[0]}) "
                    f"and labels ({len(labels)}) that cannot be rectified with ignore_indices. "
                    f"(topic_dist_rows > labels_len: {topic_distribution_np.shape[0] > len(labels)}, "
                    f"ignore_indices_len: {len(ignore_indices)})"
                )

        assert topic_distribution_np.shape[0] == len(labels), \
            f"topic-document-matrix shape {topic_distribution_np.shape[0]} does not match number of labels {len(labels)}"
        
        print("Computing pairwise KL divergence using PyTorch...")
        # Convert to tensor and compute KL divergence
        topic_distribution_tensor = torch.from_numpy(topic_distribution_np).float()
        kl_matrix_torch = compute_pairwise_kl_divergence_torch(topic_distribution_tensor, device)
        similarity_matrix = kl_matrix_torch.cpu().numpy() # Move back to CPU and convert to NumPy
        print("KL divergence computation complete.")

    elif args.method == 'probabilities':
        # Placeholder: Load probabilities representation
        print(f"Warning: Representation loading for method '{args.method}' is a placeholder.")
        # retrieval_representation = ... # Load/compute data
        # if retrieval_representation represents distributions, KL can be used:
        # representation_tensor = torch.from_numpy(retrieval_representation).float()
        # kl_matrix_torch = compute_pairwise_kl_divergence_torch(representation_tensor, device)
        # similarity_matrix = kl_matrix_torch.cpu().numpy()
        # Or use another similarity metric like cosine similarity:
        # from sklearn.metrics.pairwise import cosine_similarity
        # cos_sim = cosine_similarity(retrieval_representation)
        # similarity_matrix = 1 - cos_sim # Convert to distance
        # np.fill_diagonal(similarity_matrix, 0)
        print(f"Warning: Similarity computation for method '{args.method}' is a placeholder. Assigning NaN matrix.")
        # Need to determine the number of docs first if not loaded
        # For now, assuming labels give the number of docs
        n_docs_placeholder = len(labels)
        similarity_matrix = np.full((n_docs_placeholder, n_docs_placeholder), np.nan)

    elif args.method == 'hidden_states':
        # Placeholder: Load hidden_states representation
        print(f"Warning: Representation loading for method '{args.method}' is a placeholder.")
        # retrieval_representation = ... # Load/compute data (potentially needs aggregation)
        # Often cosine similarity or Euclidean distance is used for hidden states:
        # from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        # cos_sim = cosine_similarity(retrieval_representation)
        # similarity_matrix = 1 - cos_sim # Convert to distance
        # Or:
        # similarity_matrix = euclidean_distances(retrieval_representation)
        # np.fill_diagonal(similarity_matrix, 0)
        print(f"Warning: Similarity computation for method '{args.method}' is a placeholder. Assigning NaN matrix.")
        n_docs_placeholder = len(labels)
        similarity_matrix = np.full((n_docs_placeholder, n_docs_placeholder), np.nan)

    else:
        raise NotImplementedError(f"Method {args.method} not implemented")

    # Check if similarity matrix was computed
    if similarity_matrix is None:
        raise ValueError(f"Similarity matrix was not computed. Check implementation for method {args.method}.")
    # Check if it contains NaNs (indicating a placeholder was hit without full implementation)
    if np.isnan(similarity_matrix).all():
         raise NotImplementedError(f"Similarity calculation for method {args.method} resulted in all NaNs. Please implement it fully.")

    n_docs = similarity_matrix.shape[0]
    k_values = [1, 5, 10]
    max_k = max(k_values)
    
    precision_results = {k: [] for k in k_values}
    
    for i in tqdm(range(n_docs), desc="Calculating precision for each document"):
        # Get indices of documents sorted by similarity_matrix values (ascending order, smaller is better)
        retrieved_indices = np.argsort(similarity_matrix[i])
        # Remove self (which should have 0 distance/divergence and be first)
        retrieved_indices = retrieved_indices[retrieved_indices != i][:max_k]
        
        precisions = compute_precision_at_k(retrieved_indices, labels[i], labels, k_values)
        
        for k_val, precision in precisions.items(): # Renamed k to k_val to avoid conflict
            precision_results[int(k_val.split('@')[1])].append(precision)
    
    for k in k_values:
        if precision_results[k]:
            avg_precision = np.mean(precision_results[k])
            print(f"Average Precision@{k}: {avg_precision:.4f}")
        else:
            print(f"No results to average for Precision@{k} (perhaps not enough documents or k is too large).")
    return precision_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bow_corpus_path", type=str, default="data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt")
    parser.add_argument("--data_path", type=str, default="results/20_newsgroups/combined_K100/")
    # parser.add_argument("--probabilities_path", type=str, default="results/20_newsgroups/combined_K100/")
    parser.add_argument("--label_file", type=str, default="data/20_newsgroups_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt")
    parser.add_argument("--method", type=str, default='topic_distribution', choices=['topic_distribution', 'next_word_probs', 'hidden_states'])
    args = parser.parse_args()

    datasets = ['stackoverflow', 'dbpedia_14']
    label_files = ['data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt',
                   'data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/numeric_labels.txt']
    bow_corpus_paths = ['data/stackoverflow_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt',
                        'data/dbpedia_14_Llama-3.2-1B-Instruct_vocab_2000_last/bow_dataset.txt']
    baselines = ['lda', 'prodlda', 'combined', 'zeroshot', 'etm', 'bertopic']
    ours = ['Llama-3.2-1B-Instruct-CE', 'Llama-3.2-1B-Instruct-KL', 'Llama-3.2-3B-Instruct-CE', 'Llama-3.2-3B-Instruct-KL']
    topic_models = baselines + ours

    for dataset, label_file, bow_corpus_path in zip(datasets, label_files, bow_corpus_paths):
        print(f"Computing retrieval results for {dataset} dataset\n\n\n")
        args.bow_corpus_path = bow_corpus_path
        args.label_file = label_file
        for topic_model in topic_models:
            table_cell = []
            for num_topics in [25, 50, 75, 100]:
                if topic_model.startswith('Llama'):
                    llm_name = '-'.join(topic_model.split('-')[:-1])
                    loss_method = topic_model.split('-')[-1]
                    args.data_path = f"results/{dataset}/{llm_name}/{num_topics}_{loss_method}"
                else:
                    args.data_path = f"results/{dataset}/{topic_model}_K{num_topics}/"
                all_seed_results = []
                seed_dirs = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
                # Store results from all seeds
                all_results = defaultdict(list)
                for seed_dir in seed_dirs:
                    print(f"\nProcessing seed directory: {seed_dir}")
                    args.data_file = os.path.join(args.data_path, seed_dir, 'model_output.pt')
                    results = main(args)
                    for k, precisions in results.items():
                        all_results[k].append(np.mean(precisions))

                # Calculate and print average across all seeds
                if all_results:
                    print("\n" + "="*50)
                    print(f"AVERAGE RESULTS ACROSS {len(seed_dirs)} SEEDS for {topic_model} with {num_topics} topics")
                    print(f"Data Path: {args.data_path}")
                    print("="*50)
                    
                    for k in sorted(all_results.keys()):
                        avg_precision = np.mean(all_results[k])
                        std_precision = np.std(all_results[k])
                        table_cell.append(avg_precision)
                        print(f"Average Precision@{k}: {avg_precision:.4f} ± {std_precision:.4f}")
                else:
                    print("No results to average (perhaps not enough documents or k is too large).")
            print(f"Table row for {topic_model} with topics: [25, 50, 75, 100] on {dataset} dataset")
            print('\t'.join(map(str, table_cell)))