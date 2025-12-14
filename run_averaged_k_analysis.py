#!/usr/bin/env python
"""
Run statistical analysis with K-averaged results for all datasets.
"""

import subprocess
import sys

def run_analysis(dataset, strategy='averaged'):
    """Run analysis for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset} with strategy: {strategy}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "statistical_analysis_averaged_k.py",
        "--dataset", dataset,
        "--strategy", strategy,
        "--output_dir", "statistical_analysis_results"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {dataset} analysis complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error analyzing {dataset}: {e}")
        return False

def main():
    print("="*80)
    print("Statistical Analysis: Results Averaged Across K Values")
    print("="*80)
    
    datasets = ["dbpedia_14", "20_newsgroups", "stackoverflow"]
    
    print("\nThis will analyze performance averaged across K ∈ {25, 50, 75, 100}")
    print("Using 'averaged' strategy: Average K first, then test (n=5 per method)")
    print()
    
    results = {}
    for dataset in datasets:
        results[dataset] = run_analysis(dataset, strategy='averaged')
    
    # Print summary
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")
    
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset}")
    
    print(f"\nResults saved in: statistical_analysis_results/")
    print("\nGenerated files for each dataset:")
    print("  - {dataset}_averaged_K_metrics_averaged.csv")
    print("  - {dataset}_averaged_K_summary_averaged.json")
    print("  - {dataset}_averaged_K_summary_averaged.md  ← Read this!")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())

