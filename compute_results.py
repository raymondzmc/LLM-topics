import os
import json
from collections import defaultdict
from utils.metrics import compute_aggregate_results
import pdb


results_dir = 'results/dbpedia/generative_K50_sparsity2e-3'
averaged_results = compute_aggregate_results(results_dir)
with open(os.path.join(results_dir, 'averaged_results.json'), 'w', encoding='utf-8') as f:
    json.dump(averaged_results, f)


# ckpt_dir = 'data/bbc/bbc_news_full_Llama-3.2-1B_2000_last/checkpoints/zeroshot_K50'
# run_dirs = os.listdir(ckpt_dir)
# # Group runs by config, ignoring seed
# config_groups = defaultdict(list)
# for run_dir in run_dirs:
#     args_path = os.path.join(ckpt_dir, run_dir, 'args.json')
#     if os.path.exists(args_path):
#         with open(args_path, 'r') as f:
#             args = json.load(f)
#             # Create config key excluding seed
#             args_copy = args.copy()
#             args_copy.pop('seed', None)
#             config_key = json.dumps(args_copy, sort_keys=True)
#             config_groups[config_key].append(run_dir)

# # Only keep runs that have evaluation results
# config_groups = {k: [r for r in runs if os.path.exists(os.path.join(ckpt_dir, r, 'evaluation_results.json'))]
#                 for k, runs in config_groups.items()}

# # Remove empty groups
# config_groups = {k: v for k, v in config_groups.items() if v}

# if not config_groups:
#     print("No valid run groups found")
#     exit(1)

# for config_key, runs in config_groups.items():
#     results = defaultdict(float)
#     for run_dir in runs:
#         with open(os.path.join(ckpt_dir, run_dir, 'evaluation_results.json'), 'r') as f:
#             result = json.load(f)
#             for k, v in result.items():
#                 results[k] += v
#     print(f"Results for {config_key}:")

#     # Average results using the number of runs in this config group
#     for k in results:
#         results[k] /= len(runs)
#         print(f'{k}: {results[k]:.4f}'.replace('0.', '.'))