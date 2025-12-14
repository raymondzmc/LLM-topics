#!/usr/bin/env python
"""
Quick test script to validate the statistical analysis setup.
"""

import os
import json

def test_file_structure():
    """Test that expected files and directories exist."""
    print("Testing file structure...")
    
    base_dir = "/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results"
    
    # Test datasets exist
    datasets = ["dbpedia_14", "20_newsgroups", "stackoverflow"]
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.exists(dataset_path):
            print(f"  ✓ {dataset} directory found")
        else:
            print(f"  ✗ {dataset} directory NOT found")
            return False
    
    # Test a few specific method directories
    test_cases = [
        ("20_newsgroups", "lda_K25"),
        ("20_newsgroups", "bertopic_K50"),
        ("dbpedia_14", "combined_K75"),
    ]
    
    for dataset, method_dir in test_cases:
        full_path = os.path.join(base_dir, dataset, method_dir)
        if os.path.exists(full_path):
            print(f"  ✓ {dataset}/{method_dir} found")
            
            # Check for seed directories
            seed_0 = os.path.join(full_path, "seed_0", "evaluation_results.json")
            if os.path.exists(seed_0):
                print(f"    ✓ seed_0 results found")
            else:
                print(f"    ✗ seed_0 results NOT found")
        else:
            print(f"  ✗ {dataset}/{method_dir} NOT found")
    
    print("✓ File structure test passed\n")
    return True

def test_load_sample_result():
    """Test loading a sample result file."""
    print("Testing result file loading...")
    
    sample_file = "/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results/20_newsgroups/lda_K25/seed_0/evaluation_results.json"
    
    if not os.path.exists(sample_file):
        print(f"  ✗ Sample file not found: {sample_file}")
        return False
    
    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        print(f"  ✓ Successfully loaded JSON")
        
        # Check for expected metrics
        expected_metrics = ['llm_rating', 'npmi', 'inverted_rbo', 'purity']
        found_metrics = []
        missing_metrics = []
        
        for metric in expected_metrics:
            if metric in data:
                found_metrics.append(metric)
                print(f"    ✓ {metric}: {data[metric]:.4f}")
            else:
                missing_metrics.append(metric)
                print(f"    ✗ {metric}: NOT FOUND")
        
        if missing_metrics:
            print(f"  ⚠ Warning: Missing metrics: {missing_metrics}")
        else:
            print(f"  ✓ All expected metrics found")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return False

def test_fastopic_location():
    """Test FASTopic results location."""
    print("\nTesting FASTopic location...")
    
    fastopic_path = "/Users/raymond/Research/LLM-topics/results/20_newsgroups/fastopic_K25"
    
    if os.path.exists(fastopic_path):
        print(f"  ✓ FASTopic directory found at: {fastopic_path}")
        
        # Check for results
        seed_0 = os.path.join(fastopic_path, "seed_0", "evaluation_results.json")
        if os.path.exists(seed_0):
            print(f"    ✓ FASTopic results found")
            
            try:
                with open(seed_0, 'r') as f:
                    data = json.load(f)
                print(f"    ✓ FASTopic results loadable")
                print(f"       Sample metrics: {list(data.keys())[:5]}")
            except Exception as e:
                print(f"    ✗ Error loading FASTopic results: {e}")
        else:
            print(f"    ✗ FASTopic results NOT found")
    else:
        print(f"  ⚠ FASTopic directory not found (may not be available for all datasets)")
    
    return True

def test_llama_structure():
    """Test Llama model results structure."""
    print("\nTesting Llama model structure...")
    
    llama_path = "/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results/20_newsgroups/Llama-3.2-1B-Instruct"
    
    if os.path.exists(llama_path):
        print(f"  ✓ Llama-3.2-1B-Instruct directory found")
        
        # Check for K_KL directories
        test_dirs = ["25_KL", "50_KL", "75_KL", "100_KL"]
        for test_dir in test_dirs:
            full_path = os.path.join(llama_path, test_dir)
            if os.path.exists(full_path):
                print(f"    ✓ {test_dir} found")
            else:
                print(f"    ✗ {test_dir} NOT found")
    else:
        print(f"  ✗ Llama directory NOT found")
        return False
    
    return True

def count_available_methods():
    """Count how many methods have complete data for each dataset."""
    print("\nCounting available methods...")
    
    base_dir = "/Users/raymond/Research/LLM-topics/results/Topic-Modeling-Results"
    datasets = ["dbpedia_14", "20_newsgroups", "stackoverflow"]
    k_values = [25, 50, 75, 100]
    
    methods_to_check = [
        ('lda', lambda d, k: f"lda_K{k}"),
        ('prodlda', lambda d, k: f"prodlda_K{k}"),
        ('combined', lambda d, k: f"combined_K{k}"),
        ('zeroshot', lambda d, k: f"zeroshot_K{k}"),
        ('etm', lambda d, k: f"etm_K{k}"),
        ('bertopic', lambda d, k: f"bertopic_K{k}"),
        ('Llama-3.2-1B-Instruct', lambda d, k: f"Llama-3.2-1B-Instruct/{k}_KL"),
        ('Llama-3.2-3B-Instruct', lambda d, k: f"Llama-3.2-3B-Instruct/{k}_KL"),
    ]
    
    for dataset in datasets:
        print(f"\n  {dataset}:")
        for method_name, path_func in methods_to_check:
            complete_count = 0
            for k in k_values:
                method_path = os.path.join(base_dir, dataset, path_func(dataset, k))
                
                # Check if all 5 seeds exist
                all_seeds_exist = True
                for seed in range(5):
                    eval_file = os.path.join(method_path, f"seed_{seed}", "evaluation_results.json")
                    if not os.path.exists(eval_file):
                        all_seeds_exist = False
                        break
                
                if all_seeds_exist:
                    complete_count += 1
            
            if complete_count == 4:
                print(f"    ✓ {method_name:30s}: {complete_count}/4 K values complete")
            else:
                print(f"    ⚠ {method_name:30s}: {complete_count}/4 K values complete")

def main():
    print("="*80)
    print("Statistical Analysis Test Suite")
    print("="*80)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Sample Result Loading", test_load_sample_result),
        ("FASTopic Location", test_fastopic_location),
        ("Llama Structure", test_llama_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            failed += 1
        print()
    
    # Always run method counting (informational)
    count_available_methods()
    
    print("\n" + "="*80)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n✓ All tests passed! You can now run the statistical analysis.")
        print("\nRun: python run_all_analyses.py")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
