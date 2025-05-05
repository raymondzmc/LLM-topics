import os
import json
import re
import argparse
import collections

# Define metric keys globally
# REQUIRED_KEYS = ['npmi', 'word2vec_word_embeddings', 'llm_rating', 'topic_diversity', 'inverted_rbo']
REQUIRED_KEYS = ['npmi', 'word2vec_word_embeddings', 'llm_rating', 'topic_diversity', 'inverted_rbo', 'harmonic_purity', 'ari', 'mis']
NUM_METRICS = len(REQUIRED_KEYS)

def get_metrics(file_path):
    """Reads metrics from a JSON file and returns a list of formatted strings."""
    # Initialize with N/A for all metrics
    metric_values = ["N/A"] * NUM_METRICS
    try:
        if not os.path.exists(file_path):
            return metric_values # Return list of N/A
        with open(file_path, 'r') as f:
            results = json.load(f)

        # Check if all required keys are present
        if not all(key in results for key in REQUIRED_KEYS):
             print(f"Warning: Missing keys in {file_path}. Found: {list(results.keys())}")
             # Try to get values for keys that *are* present
             formatted_values = []
             for key in REQUIRED_KEYS:
                 if key in results:
                     try:
                        formatted_values.append(f"{float(results[key]):.4f}")
                     except (ValueError, TypeError):
                         formatted_values.append("Invalid") # Handle non-numeric data
                 else:
                     formatted_values.append("N/A")
             return formatted_values

        # Format numbers if all keys are present
        return [f"{results[key]:.4f}" for key in REQUIRED_KEYS]

    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}")
        return metric_values # Return list of N/A
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
        return metric_values # Return list of N/A

def main(args):
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
        
    all_items = os.listdir(results_dir)

    # --- Identify all K values ---
    k_values = set()
    # Pattern for simple methods like lda_K25, combined_K50
    simple_method_pattern = re.compile(r'^([a-zA-Z0-9_-]+)_K(\d+)$')
    # Pattern for Llama subdirs like 25_CE, 100_KL
    llama_subdir_pattern = re.compile(r'^(\d+)_(CE|KL)$')

    for item_name in all_items:
        full_path = os.path.join(results_dir, item_name)
        if not os.path.isdir(full_path):
            continue

        # Check simple methods
        simple_match = simple_method_pattern.match(item_name)
        if simple_match:
            k = int(simple_match.group(2))
            k_values.add(k)
            continue # Move to next item

        # Check Llama-style methods by looking inside their directories
        if item_name.startswith("Llama"):
             llama_path = full_path
             if os.path.isdir(llama_path):
                 for sub_item in os.listdir(llama_path):
                     sub_item_path = os.path.join(llama_path, sub_item)
                     if not os.path.isdir(sub_item_path):
                         continue
                     llama_sub_match = llama_subdir_pattern.match(sub_item)
                     if llama_sub_match:
                         k = int(llama_sub_match.group(1))
                         k_values.add(k)

    sorted_ks = sorted(list(k_values))
    if not sorted_ks:
        print(f"Error: No K values found in directory structure under {results_dir}")
        return

    # --- Initialize results structure ---
    # Use defaultdict for easy initialization
    methods_results = collections.defaultdict(lambda: {k: ["N/A"] * NUM_METRICS for k in sorted_ks})

    # --- Populate results ---
    for item_name in all_items:
        full_path = os.path.join(results_dir, item_name)
        if not os.path.isdir(full_path):
            continue

        # Process simple methods
        simple_match = simple_method_pattern.match(item_name)
        if simple_match:
            method_name = simple_match.group(1)
            k = int(simple_match.group(2))
            results_path = os.path.join(full_path, 'averaged_results.json')
            metrics = get_metrics(results_path)
            if k in methods_results[method_name]: # Ensure K was found initially
                 methods_results[method_name][k] = metrics # metrics is now a list
            continue # Move to next item

        # Process Llama-style methods
        if item_name.startswith("Llama"):
            llama_path = full_path
            if os.path.isdir(llama_path):
                 for sub_item in os.listdir(llama_path):
                    sub_item_path = os.path.join(llama_path, sub_item)
                    if not os.path.isdir(sub_item_path):
                        continue

                    llama_sub_match = llama_subdir_pattern.match(sub_item)
                    if llama_sub_match:
                        k = int(llama_sub_match.group(1))
                        technique = llama_sub_match.group(2) # CE or KL
                        # Combine base Llama model name with technique
                        method_name = f"{item_name}-{technique}" 
                        
                        results_path = os.path.join(sub_item_path, 'averaged_results.json')
                        metrics = get_metrics(results_path)
                        if k in methods_results[method_name]: # Ensure K was found initially
                            methods_results[method_name][k] = metrics # metrics is now a list

    # --- Print results ---
    # Sort methods alphabetically for consistent output
    sorted_methods = sorted(methods_results.keys())

    # Print header Row 1 (K values spanning metric columns)
    header1_parts = ["Method"]
    for k in sorted_ks:
        # Add K value, followed by tabs to span the metric columns
        header1_parts.append(f"K{k}" + "\t" * (NUM_METRICS - 1))
    print("\t".join(header1_parts))

    # Print header Row 2 (Metric names repeated under each K)
    header2_parts = [""] # Empty cell for 'Method' column
    for _ in sorted_ks:
        header2_parts.append("\t".join(REQUIRED_KEYS))
    print("\t".join(header2_parts))


    # Print data rows
    for method in sorted_methods:
        # Fetch results lists for each K in the sorted order
        row_data_lists = [methods_results[method].get(k, ["N/A"] * NUM_METRICS) for k in sorted_ks]
        # Flatten the list of lists into a single list for the row
        flat_row_data = [item for sublist in row_data_lists for item in sublist]
        # Combine method name and flattened data
        row = [method]
        row.extend(flat_row_data)
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and print topic modeling results based on directory structure.")
    parser.add_argument("--results_dir", type=str, default="results/20_newsgroups",
                        help="Directory containing the result subdirectories.")
    args = parser.parse_args()
    # Removed old logic for finding num_topics here
    main(args)
