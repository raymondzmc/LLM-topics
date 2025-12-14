#!/usr/bin/env python
"""
Statistical significance testing with results averaged across K values.
This computes significance when averaging performance over K ∈ {25, 50, 75, 100}.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import argparse


def load_results(results_dir: str, dataset: str, methods: List[str], k_values: List[int]) -> Dict:
    """Load evaluation results for all methods, K values, and seeds."""
    all_results = {}
    parent_results_dir = os.path.dirname(results_dir)
    
    for method in methods:
        all_results[method] = {}
        for k in k_values:
            seed_results = []
            
            if method in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct']:
                method_dir = os.path.join(results_dir, dataset, method, f"{k}_KL")
            elif method == 'fastopic':
                method_dir = os.path.join(parent_results_dir, dataset, f"fastopic_K{k}")
            else:
                method_dir = os.path.join(results_dir, dataset, f"{method}_K{k}")
            
            if not os.path.exists(method_dir):
                continue
            
            for seed in range(5):
                seed_dir = os.path.join(method_dir, f"seed_{seed}")
                eval_file = os.path.join(seed_dir, "evaluation_results.json")
                
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        result = json.load(f)
                        seed_results.append(result)
            
            if seed_results:
                all_results[method][k] = seed_results
    
    return all_results


def compute_averaged_metrics(results: Dict, metrics: List[str]) -> pd.DataFrame:
    """
    Compute metrics averaged across K values.
    
    Two averaging strategies:
    1. Average across K first, then we have 5 values per method (one per seed)
    2. Treat each K×seed combination as independent (20 values per method)
    
    This function implements Strategy 1 (recommended for main table).
    """
    rows = []
    
    for method, k_dict in results.items():
        if not k_dict:  # Skip if no results
            continue
        
        # Get all K values available for this method
        available_k = sorted(k_dict.keys())
        
        # For each seed, average across K values
        num_seeds = len(k_dict[available_k[0]]) if available_k else 0
        
        for seed_idx in range(num_seeds):
            row = {'method': method, 'seed': seed_idx}
            
            for metric in metrics:
                # Collect metric values across all K for this seed
                metric_values = []
                for k in available_k:
                    if seed_idx < len(k_dict[k]):
                        value = k_dict[k][seed_idx].get(metric, np.nan)
                        if not np.isnan(value):
                            metric_values.append(value)
                
                # Average across K values
                if metric_values:
                    row[metric] = np.mean(metric_values)
                else:
                    row[metric] = np.nan
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def compute_expanded_metrics(results: Dict, metrics: List[str]) -> pd.DataFrame:
    """
    Compute metrics treating each K×seed as independent observation.
    
    This gives 20 observations per method (5 seeds × 4 K values).
    More conservative approach.
    """
    rows = []
    
    for method, k_dict in results.items():
        for k, seed_results in k_dict.items():
            for seed_idx, result in enumerate(seed_results):
                row = {'method': method, 'K': k, 'seed': seed_idx}
                for metric in metrics:
                    row[metric] = result.get(metric, np.nan)
                rows.append(row)
    
    return pd.DataFrame(rows)


def perform_statistical_tests(df: pd.DataFrame, metric: str, strategy: str = 'averaged') -> Tuple[Dict, Dict]:
    """
    Perform pairwise t-tests between methods.
    
    Args:
        df: DataFrame with method, seed, and metric columns
        metric: Metric to analyze
        strategy: 'averaged' (K averaged first) or 'expanded' (all K×seed)
    """
    methods = df['method'].unique()
    
    # Compute statistics for each method
    stats_dict = {}
    for method in methods:
        method_data = df[df['method'] == method][metric].values
        method_data = method_data[~np.isnan(method_data)]  # Remove NaN values
        
        if len(method_data) > 0:
            stats_dict[method] = {
                'mean': np.mean(method_data),
                'std': np.std(method_data, ddof=1) if len(method_data) > 1 else 0,
                'n': len(method_data),
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
            
            if len(data1) > 1 and len(data2) > 1:
                # Use Welch's t-test (doesn't assume equal variances)
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                pairwise_pvalues[f"{method1}_vs_{method2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'n1': len(data1),
                    'n2': len(data2)
                }
    
    return stats_dict, pairwise_pvalues


def find_best_methods(stats_dict: Dict, pairwise_pvalues: Dict, 
                      alpha: float = 0.05, higher_is_better: bool = True) -> List[str]:
    """Identify statistically best method(s)."""
    if not stats_dict:
        return []
    
    sorted_methods = sorted(stats_dict.items(), 
                          key=lambda x: x[1]['mean'], 
                          reverse=higher_is_better)
    
    best_methods = [sorted_methods[0][0]]
    
    for method, method_stats in sorted_methods[1:]:
        comparison_key = f"{best_methods[0]}_vs_{method}"
        reverse_key = f"{method}_vs_{best_methods[0]}"
        
        is_significantly_different = False
        if comparison_key in pairwise_pvalues:
            p_val = pairwise_pvalues[comparison_key]['p_value']
            is_significantly_different = p_val < alpha
        elif reverse_key in pairwise_pvalues:
            p_val = pairwise_pvalues[reverse_key]['p_value']
            is_significantly_different = p_val < alpha
        
        if not is_significantly_different:
            best_methods.append(method)
    
    return best_methods


def analyze_averaged_results(results_dir: str, dataset: str, output_dir: str, strategy: str = 'averaged'):
    """Main analysis function for averaged results."""
    
    methods = [
        'lda', 'prodlda', 'combined', 'zeroshot', 'etm', 'bertopic', 
        'fastopic', 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct'
    ]
    k_values = [25, 50, 75, 100]
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    higher_is_better = {
        'llm_rating': True,
        'npmi': True,
        'inverted_rbo': True,
        'purity': True,
        'word2vec_word_embeddings': True
    }
    
    print(f"Loading results for dataset: {dataset}")
    all_results = load_results(results_dir, dataset, methods, k_values)
    
    print(f"Computing metrics with strategy: {strategy}")
    if strategy == 'averaged':
        # Average across K first (5 observations per method)
        df = compute_averaged_metrics(all_results, metrics)
        print(f"Using averaged approach: {len(df)} total observations")
    else:  # expanded
        # Treat each K×seed as independent (up to 20 observations per method)
        df = compute_expanded_metrics(all_results, metrics)
        print(f"Using expanded approach: {len(df)} total observations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    csv_filename = f'{dataset}_averaged_K_metrics_{strategy}.csv'
    df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    print(f"Saved raw metrics to {os.path.join(output_dir, csv_filename)}")
    
    # Analyze each metric
    results_summary = {}
    
    print(f"\n{'='*80}")
    print(f"Statistical Analysis: {dataset} (averaged across K values)")
    print(f"Strategy: {strategy}")
    print(f"{'='*80}\n")
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"Metric: {metric}")
        print("-" * 40)
        
        stats_dict, pairwise_pvalues = perform_statistical_tests(df, metric, strategy)
        
        if not stats_dict:
            print(f"No data available for {metric}")
            print()
            continue
        
        # Print statistics
        for method, stats_info in sorted(stats_dict.items(), 
                                        key=lambda x: x[1]['mean'], 
                                        reverse=higher_is_better[metric]):
            n = stats_info['n']
            print(f"{method:30s}: {stats_info['mean']:.4f} ± {stats_info['std']:.4f} (n={n})")
        
        # Find best methods
        best_methods = find_best_methods(stats_dict, pairwise_pvalues, 
                                        alpha=0.05, 
                                        higher_is_better=higher_is_better[metric])
        
        print(f"\nBest method(s) (statistically): {', '.join(best_methods)}")
        print()
        
        # Store results
        results_summary[metric] = {
            'statistics': {m: {'mean': s['mean'], 'std': s['std'], 'n': s['n']} 
                         for m, s in stats_dict.items()},
            'best_methods': best_methods,
            'pairwise_tests': pairwise_pvalues
        }
    
    # Save summary
    json_filename = f'{dataset}_averaged_K_summary_{strategy}.json'
    with open(os.path.join(output_dir, json_filename), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create summary table
    create_summary_table(results_summary, dataset, output_dir, strategy)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_dir}")
    print(f"{'='*80}\n")
    
    return results_summary


def create_summary_table(results_summary: Dict, dataset: str, output_dir: str, strategy: str):
    """Create markdown summary table."""
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    table_lines = []
    table_lines.append(f"# Statistical Analysis Summary: {dataset} (Averaged across K values)")
    table_lines.append(f"## Strategy: {strategy}")
    table_lines.append("")
    table_lines.append("## Best Methods by Metric (Averaged across K ∈ {25, 50, 75, 100})")
    table_lines.append("")
    
    # Header
    table_lines.append("| Metric | Best Method(s) | Mean ± Std |")
    table_lines.append("|--------|----------------|------------|")
    
    # Rows
    for metric in metrics:
        if metric not in results_summary:
            continue
        
        best_methods = results_summary[metric]['best_methods']
        stats = results_summary[metric]['statistics']
        
        if best_methods:
            best_method = best_methods[0]
            mean = stats[best_method]['mean']
            std = stats[best_method]['std']
            
            if len(best_methods) > 1:
                methods_str = f"{best_methods[0]} (+{len(best_methods)-1} tied)"
            else:
                methods_str = best_methods[0]
            
            table_lines.append(f"| {metric} | {methods_str} | {mean:.4f} ± {std:.4f} |")
    
    table_lines.append("")
    table_lines.append("## Detailed Rankings")
    table_lines.append("")
    
    for metric in metrics:
        if metric not in results_summary:
            continue
        
        table_lines.append(f"### {metric}")
        table_lines.append("")
        
        stats = results_summary[metric]['statistics']
        sorted_methods = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        table_lines.append("| Rank | Method | Mean | Std | n |")
        table_lines.append("|------|--------|------|-----|---|")
        
        for rank, (method, method_stats) in enumerate(sorted_methods, 1):
            mean = method_stats['mean']
            std = method_stats['std']
            n = method_stats['n']
            marker = " ⭐" if method in results_summary[metric]['best_methods'] else ""
            table_lines.append(f"| {rank} | {method}{marker} | {mean:.4f} | {std:.4f} | {n} |")
        
        table_lines.append("")
    
    # Save
    md_filename = f'{dataset}_averaged_K_summary_{strategy}.md'
    with open(os.path.join(output_dir, md_filename), 'w') as f:
        f.write("\n".join(table_lines))
    
    print(f"Summary table saved to {os.path.join(output_dir, md_filename)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistical analysis with results averaged across K values'
    )
    parser.add_argument('--results_dir', type=str, 
                       default='/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results',
                       help='Directory containing topic modeling results')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['dbpedia_14', '20_newsgroups', 'stackoverflow'],
                       help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, 
                       default='statistical_analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--strategy', type=str, default='averaged',
                       choices=['averaged', 'expanded'],
                       help='averaged: average K first (n=5 per method), '
                            'expanded: treat K×seed as independent (n=20 per method)')
    
    args = parser.parse_args()
    
    analyze_averaged_results(args.results_dir, args.dataset, args.output_dir, args.strategy)

