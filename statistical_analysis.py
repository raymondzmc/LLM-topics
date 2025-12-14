import os
import json
import numpy as np
from typing import Dict, List, Tuple
import argparse

try:
    import pandas as pd
    from scipy import stats
except ImportError:
    print("ERROR: Missing dependencies. Please install:")
    print("  pip install pandas scipy")
    exit(1)


def load_results(results_dir: str, dataset: str, methods: List[str], k_values: List[int]) -> Dict:
    """
    Load evaluation results for all methods, K values, and seeds.
    
    Returns:
        Dict with structure: {method: {k: [seed_results]}}
    """
    all_results = {}
    
    # Get parent directory for FASTopic results
    parent_results_dir = os.path.dirname(results_dir)
    
    for method in methods:
        all_results[method] = {}
        for k in k_values:
            seed_results = []
            
            # Handle different directory naming conventions
            if method in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct']:
                # For Llama models, use KL loss variant
                method_dir = os.path.join(results_dir, dataset, method, f"{k}_KL")
            elif method == 'fastopic':
                # FASTopic is in the parent results directory
                method_dir = os.path.join(parent_results_dir, dataset, f"fastopic_K{k}")
            else:
                method_dir = os.path.join(results_dir, dataset, f"{method}_K{k}")
            
            if not os.path.exists(method_dir):
                print(f"Warning: Directory not found: {method_dir}")
                continue
            
            # Load results from each seed
            for seed in range(5):
                seed_dir = os.path.join(method_dir, f"seed_{seed}")
                eval_file = os.path.join(seed_dir, "evaluation_results.json")
                
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        result = json.load(f)
                        seed_results.append(result)
                else:
                    print(f"Warning: File not found: {eval_file}")
            
            if seed_results:
                all_results[method][k] = seed_results
    
    return all_results


def extract_metrics(results: Dict, metrics: List[str]) -> pd.DataFrame:
    """
    Extract specific metrics from results and organize into a DataFrame.
    
    Returns:
        DataFrame with columns: [method, K, seed, metric1, metric2, ...]
    """
    rows = []
    
    for method, k_dict in results.items():
        for k, seed_results in k_dict.items():
            for seed_idx, result in enumerate(seed_results):
                row = {
                    'method': method,
                    'K': k,
                    'seed': seed_idx
                }
                for metric in metrics:
                    row[metric] = result.get(metric, np.nan)
                rows.append(row)
    
    return pd.DataFrame(rows)


def compute_statistical_tests(df: pd.DataFrame, metric: str, k: int) -> Dict:
    """
    Perform statistical significance tests between all methods for a given metric and K.
    
    Returns:
        Dictionary with mean, std, and p-values from pairwise t-tests
    """
    # Filter for specific K value
    df_k = df[df['K'] == k].copy()
    
    # Get unique methods
    methods = df_k['method'].unique()
    
    # Compute statistics for each method
    stats_dict = {}
    for method in methods:
        method_data = df_k[df_k['method'] == method][metric].values
        if len(method_data) > 0:
            stats_dict[method] = {
                'mean': np.mean(method_data),
                'std': np.std(method_data, ddof=1) if len(method_data) > 1 else 0,
                'values': method_data
            }
    
    # Perform pairwise t-tests
    pairwise_pvalues = {}
    for i, method1 in enumerate(methods):
        if method1 not in stats_dict:
            continue
        for method2 in methods[i+1:]:
            if method2 not in stats_dict:
                continue
            
            data1 = stats_dict[method1]['values']
            data2 = stats_dict[method2]['values']
            
            # Perform two-sample t-test
            if len(data1) > 1 and len(data2) > 1:
                t_stat, p_value = stats.ttest_ind(data1, data2)
                pairwise_pvalues[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value
                }
    
    return stats_dict, pairwise_pvalues


def find_best_methods(stats_dict: Dict, pairwise_pvalues: Dict, 
                      alpha: float = 0.05, higher_is_better: bool = True) -> List[str]:
    """
    Identify the best method(s) based on mean performance and statistical significance.
    
    A method is considered "best" if:
    1. It has the highest mean (or tied for highest)
    2. It is not significantly worse than any other method
    """
    if not stats_dict:
        return []
    
    # Sort methods by mean performance
    sorted_methods = sorted(stats_dict.items(), 
                          key=lambda x: x[1]['mean'], 
                          reverse=higher_is_better)
    
    best_methods = [sorted_methods[0][0]]
    best_mean = sorted_methods[0][1]['mean']
    
    # Check for methods that are not significantly different from the best
    for method, method_stats in sorted_methods[1:]:
        # Check if this method is significantly different from the best method
        comparison_key = f"{best_methods[0]}_vs_{method}"
        reverse_key = f"{method}_vs_{best_methods[0]}"
        
        is_significantly_different = False
        if comparison_key in pairwise_pvalues:
            p_val = pairwise_pvalues[comparison_key]['p_value']
            is_significantly_different = p_val < alpha
        elif reverse_key in pairwise_pvalues:
            p_val = pairwise_pvalues[reverse_key]['p_value']
            is_significantly_different = p_val < alpha
        
        # If not significantly different from the best, include it
        if not is_significantly_different:
            best_methods.append(method)
    
    return best_methods


def run_analysis(results_dir: str, dataset: str, output_dir: str):
    """
    Main analysis function.
    """
    # Define methods and K values
    methods = [
        'lda', 'prodlda', 'combined', 'zeroshot', 'etm', 'bertopic', 
        'fastopic', 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct'
    ]
    k_values = [25, 50, 75, 100]
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    # Metrics where higher is better
    higher_is_better = {
        'llm_rating': True,
        'npmi': True,
        'inverted_rbo': True,
        'purity': True,
        'word2vec_word_embeddings': True
    }
    
    print(f"Loading results for dataset: {dataset}")
    all_results = load_results(results_dir, dataset, methods, k_values)
    
    print("Extracting metrics into DataFrame...")
    df = extract_metrics(all_results, metrics)
    
    # Save raw data
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f'{dataset}_all_metrics.csv'), index=False)
    print(f"Saved raw metrics to {os.path.join(output_dir, f'{dataset}_all_metrics.csv')}")
    
    # Analyze for each K and metric
    results_summary = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"Analyzing K={k}")
        print(f"{'='*80}")
        
        results_summary[k] = {}
        
        for metric in metrics:
            print(f"\nMetric: {metric}")
            print("-" * 40)
            
            stats_dict, pairwise_pvalues = compute_statistical_tests(df, metric, k)
            
            if not stats_dict:
                print(f"No data available for {metric} at K={k}")
                continue
            
            # Print statistics for all methods
            for method, stats_info in sorted(stats_dict.items(), 
                                            key=lambda x: x[1]['mean'], 
                                            reverse=higher_is_better[metric]):
                print(f"{method:30s}: {stats_info['mean']:.4f} ± {stats_info['std']:.4f}")
            
            # Find best methods
            best_methods = find_best_methods(stats_dict, pairwise_pvalues, 
                                            alpha=0.05, 
                                            higher_is_better=higher_is_better[metric])
            
            print(f"\nBest method(s) (statistically): {', '.join(best_methods)}")
            
            # Store results
            results_summary[k][metric] = {
                'statistics': {m: {'mean': s['mean'], 'std': s['std']} 
                             for m, s in stats_dict.items()},
                'best_methods': best_methods,
                'pairwise_tests': pairwise_pvalues
            }
    
    # Save summary results
    with open(os.path.join(output_dir, f'{dataset}_statistical_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_dir}")
    print(f"{'='*80}")
    
    # Create a summary table
    create_summary_table(results_summary, dataset, output_dir)
    
    return results_summary


def create_summary_table(results_summary: Dict, dataset: str, output_dir: str):
    """
    Create a human-readable summary table showing the best method for each K and metric.
    """
    k_values = sorted(results_summary.keys())
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    # Create markdown table
    table_lines = []
    table_lines.append(f"# Statistical Analysis Summary: {dataset}")
    table_lines.append("")
    table_lines.append("## Best Methods by K and Metric")
    table_lines.append("")
    
    # Header
    header = "| K | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics) + 1)) + "|"
    table_lines.append(header)
    table_lines.append(separator)
    
    # Rows
    for k in k_values:
        row = f"| {k} |"
        for metric in metrics:
            if metric in results_summary[k]:
                best_methods = results_summary[k][metric]['best_methods']
                stats = results_summary[k][metric]['statistics']
                
                # Get mean for best method
                best_method_info = []
                for method in best_methods:
                    mean = stats[method]['mean']
                    std = stats[method]['std']
                    best_method_info.append(f"{method} ({mean:.3f}±{std:.3f})")
                
                cell = "<br>".join(best_method_info[:3])  # Limit to top 3
                if len(best_method_info) > 3:
                    cell += "<br>..."
            else:
                cell = "N/A"
            row += f" {cell} |"
        table_lines.append(row)
    
    # Add detailed statistics
    table_lines.append("")
    table_lines.append("## Detailed Statistics")
    table_lines.append("")
    
    for k in k_values:
        table_lines.append(f"### K = {k}")
        table_lines.append("")
        
        for metric in metrics:
            if metric not in results_summary[k]:
                continue
                
            table_lines.append(f"#### {metric}")
            table_lines.append("")
            
            stats = results_summary[k][metric]['statistics']
            sorted_methods = sorted(stats.items(), 
                                  key=lambda x: x[1]['mean'], 
                                  reverse=True)
            
            table_lines.append("| Rank | Method | Mean | Std |")
            table_lines.append("|------|--------|------|-----|")
            
            for rank, (method, method_stats) in enumerate(sorted_methods, 1):
                mean = method_stats['mean']
                std = method_stats['std']
                marker = " ⭐" if method in results_summary[k][metric]['best_methods'] else ""
                table_lines.append(f"| {rank} | {method}{marker} | {mean:.4f} | {std:.4f} |")
            
            table_lines.append("")
    
    # Save table
    table_path = os.path.join(output_dir, f'{dataset}_summary_table.md')
    with open(table_path, 'w') as f:
        f.write("\n".join(table_lines))
    
    print(f"\nSummary table saved to {table_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis of topic modeling results')
    parser.add_argument('--results_dir', type=str, 
                       default='/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results',
                       help='Directory containing topic modeling results')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['dbpedia_14', '20_newsgroups', 'stackoverflow'],
                       help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='statistical_analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    run_analysis(args.results_dir, args.dataset, args.output_dir)
