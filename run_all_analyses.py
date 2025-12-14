#!/usr/bin/env python
"""
Simple script to run statistical analysis for all datasets.
This avoids shell issues and can be run directly with: python run_all_analyses.py
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import pandas
        import scipy
        print("✓ Dependencies found")
        return True
    except ImportError:
        print("✗ Missing dependencies!")
        print("\nPlease install required packages:")
        print("  pip install pandas scipy")
        return False

def run_analysis(dataset):
    """Run analysis for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset}...")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "statistical_analysis.py",
        "--dataset", dataset,
        "--output_dir", "statistical_analysis_results"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {dataset} analysis complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error analyzing {dataset}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

def main():
    print("="*80)
    print("Statistical Analysis Runner")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "scipy"],
                         check=True)
            print("✓ Dependencies installed")
        except:
            print("✗ Failed to install dependencies")
            print("  Please run manually: pip install pandas scipy")
            return 1
    
    # Create output directory
    os.makedirs("statistical_analysis_results", exist_ok=True)
    
    # Run analysis for each dataset
    datasets = ["dbpedia_14", "20_newsgroups", "stackoverflow"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = run_analysis(dataset)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}\n")
    
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset}")
    
    print(f"\nResults saved in: statistical_analysis_results/")
    
    # Return 0 if all successful, 1 otherwise
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
