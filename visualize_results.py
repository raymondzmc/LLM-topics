#!/usr/bin/env python
"""
Create visualizations from statistical analysis results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from pathlib import Path

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

def load_data(results_dir, dataset):
    """Load the analysis results."""
    csv_path = os.path.join(results_dir, f'{dataset}_all_metrics.csv')
    json_path = os.path.join(results_dir, f'{dataset}_statistical_summary.json')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}\nRun statistical analysis first!")
    
    df = pd.read_csv(csv_path)
    
    with open(json_path, 'r') as f:
        summary = json.load(f)
    
    return df, summary

def create_comparison_plot(df, dataset, metric, output_dir):
    """Create a comparison plot for a specific metric across K values."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset} - {metric.upper()} Comparison', fontsize=16, fontweight='bold')
    
    k_values = [25, 50, 75, 100]
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        df_k = df[df['K'] == k].copy()
        
        # Group by method and calculate statistics
        grouped = df_k.groupby('method')[metric].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Create bar plot with error bars
        x_pos = np.arange(len(grouped))
        bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                     capsize=5, alpha=0.7, color='steelblue', 
                     edgecolor='black', linewidth=1.5)
        
        # Color the best method(s) differently
        # For this, we'd need to identify which are statistically best
        # For simplicity, just color the top performer
        if len(bars) > 0:
            bars[0].set_color('darkgreen')
            bars[0].set_alpha(0.8)
        
        ax.set_xlabel('Method', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=11, fontweight='bold')
        ax.set_title(f'K = {k}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped['method'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, (bar, mean_val) in enumerate(zip(bars, grouped['mean'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{dataset}_{metric}_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_heatmap(df, dataset, output_dir):
    """Create a heatmap showing all metrics for all methods and K values."""
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # Calculate mean for each method-K combination
        pivot_data = df.groupby(['method', 'K'])[metric].mean().reset_index()
        pivot_table = pivot_data.pivot(index='method', columns='K', values=metric)
        
        # Sort by mean performance across all K values
        pivot_table['mean'] = pivot_table.mean(axis=1)
        pivot_table = pivot_table.sort_values('mean', ascending=False)
        pivot_table = pivot_table.drop('mean', axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels([f'K={k}' for k in pivot_table.columns])
        ax.set_yticklabels(pivot_table.index)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        plt.setp(ax.get_yticklabels(), fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, rotation=270, labelpad=20, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha="center", va="center", color="black",
                                 fontsize=9, fontweight='bold')
        
        ax.set_title(f'{dataset} - {metric.upper()} Heatmap', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Topics (K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Method', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{dataset}_{metric}_heatmap.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")

def create_radar_plot(df, dataset, k_value, output_dir):
    """Create radar plot comparing methods across all metrics for a given K."""
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    
    # Filter for specific K
    df_k = df[df['K'] == k_value].copy()
    
    # Get top 5 methods by average performance
    method_scores = {}
    for method in df_k['method'].unique():
        method_df = df_k[df_k['method'] == method]
        # Normalize and average across metrics
        scores = []
        for metric in metrics:
            if metric in method_df.columns:
                scores.append(method_df[metric].mean())
        method_scores[method] = np.mean(scores) if scores else 0
    
    top_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_method_names = [m[0] for m in top_methods]
    
    # Calculate mean for each method-metric combination
    plot_data = []
    for method in top_method_names:
        method_df = df_k[df_k['method'] == method]
        values = []
        for metric in metrics:
            if metric in method_df.columns:
                values.append(method_df[metric].mean())
            else:
                values.append(0)
        plot_data.append(values)
    
    # Number of metrics
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    plot_data_wrapped = [vals + [vals[0]] for vals in plot_data]  # Complete the circle
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for different methods
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_method_names)))
    
    # Plot each method
    for idx, (method, values) in enumerate(zip(top_method_names, plot_data_wrapped)):
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    ax.set_title(f'{dataset} - Top Methods Comparison (K={k_value})',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{dataset}_radar_K{k_value}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def visualize_dataset(dataset, results_dir='statistical_analysis_results', 
                     output_dir='statistical_analysis_results/plots'):
    """Create all visualizations for a dataset."""
    print(f"\nGenerating visualizations for {dataset}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df, summary = load_data(results_dir, dataset)
    
    # Create comparison plots for each metric
    metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity', 'word2vec_word_embeddings']
    for metric in metrics:
        if metric in df.columns:
            create_comparison_plot(df, dataset, metric, output_dir)
    
    # Create heatmaps
    create_heatmap(df, dataset, output_dir)
    
    # Create radar plots for each K value
    for k in [25, 50, 75, 100]:
        if k in df['K'].values:
            create_radar_plot(df, dataset, k, output_dir)
    
    print(f"✓ Visualizations complete for {dataset}\n")

def main():
    """Main function to generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize topic modeling analysis results')
    parser.add_argument('--results_dir', type=str, default='statistical_analysis_results',
                       help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, default='statistical_analysis_results/plots',
                       help='Directory to save plots')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'dbpedia_14', '20_newsgroups', 'stackoverflow'],
                       help='Dataset to visualize (default: all)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Topic Modeling Results Visualization")
    print("="*80)
    
    datasets = ['dbpedia_14', '20_newsgroups', 'stackoverflow'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        try:
            visualize_dataset(dataset, args.results_dir, args.output_dir)
        except FileNotFoundError as e:
            print(f"✗ Skipping {dataset}: {e}")
        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
    
    print("="*80)
    print(f"Plots saved in: {args.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
