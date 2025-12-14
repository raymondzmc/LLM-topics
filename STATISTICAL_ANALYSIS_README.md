# Statistical Analysis of Topic Modeling Results

This directory contains scripts for performing statistical significance testing on topic modeling results across different methods and hyperparameters.

## Overview

The analysis compares the following methods:
- **LDA**: Latent Dirichlet Allocation
- **ProdLDA**: Product of Experts LDA
- **Combined (CWTM)**: Combined Contextualized Word-Topic Model
- **Zeroshot**: Zero-shot topic modeling
- **ETM**: Embedded Topic Model
- **BERTopic**: BERT-based topic modeling
- **FASTopic**: Fast topic modeling
- **Llama-3.2-1B-Instruct**: LLM-based topic modeling (1B parameters)
- **Llama-3.2-3B-Instruct**: LLM-based topic modeling (3B parameters)

## Metrics Analyzed

For each method and K value (25, 50, 75, 100), we analyze:

1. **LLM Rating**: Quality rating from language model evaluation (higher is better)
2. **NPMI**: Normalized Pointwise Mutual Information - coherence metric (higher is better)
3. **IRBO**: Inverted Rank-Biased Overlap - diversity metric (higher is better)
4. **Purity**: Cluster purity for classification tasks (higher is better)
5. **Word2Vec Embeddings**: Word2Vec-based embedding coherence (higher is better)

## Installation

First, install the required dependencies:

```bash
pip install pandas scipy
```

All other dependencies should already be in your requirements.txt.

## Usage

### Quick Start

Run the analysis for all datasets at once:

```bash
chmod +x run_statistical_analysis.sh
./run_statistical_analysis.sh
```

### Individual Dataset Analysis

Run analysis for a specific dataset:

```bash
# For 20 Newsgroups
python statistical_analysis.py --dataset 20_newsgroups --output_dir statistical_analysis_results

# For DBpedia
python statistical_analysis.py --dataset dbpedia_14 --output_dir statistical_analysis_results

# For StackOverflow
python statistical_analysis.py --dataset stackoverflow --output_dir statistical_analysis_results
```

### Custom Results Directory

If your results are in a different location:

```bash
python statistical_analysis.py \
    --dataset 20_newsgroups \
    --results_dir /path/to/Topic-Modeling-Results \
    --output_dir my_analysis_results
```

## Output Files

The script generates several output files in the `statistical_analysis_results/` directory:

### 1. Raw Metrics CSV
`{dataset}_all_metrics.csv`

Contains all raw metric values for each method, K value, and seed:
- Columns: method, K, seed, llm_rating, npmi, inverted_rbo, purity
- Useful for further custom analysis

### 2. Statistical Summary JSON
`{dataset}_statistical_summary.json`

Comprehensive JSON file containing:
- Mean and standard deviation for each method
- Pairwise t-test results (t-statistics and p-values)
- List of best methods based on statistical significance

Structure:
```json
{
  "25": {
    "llm_rating": {
      "statistics": {
        "method_name": {"mean": 0.xxx, "std": 0.xxx},
        ...
      },
      "best_methods": ["method1", "method2"],
      "pairwise_tests": {
        "method1_vs_method2": {
          "t_statistic": 1.234,
          "p_value": 0.045
        }
      }
    },
    ...
  },
  ...
}
```

### 3. Summary Table Markdown
`{dataset}_summary_table.md`

Human-readable markdown table showing:
- Best method(s) for each K value and metric
- Detailed statistics ranked by performance
- Methods marked with ⭐ are statistically tied for best

## Statistical Testing Methodology

### Pairwise Comparisons
- **Test**: Two-sample t-test (independent samples)
- **Null Hypothesis**: The two methods have the same mean performance
- **Significance Level**: α = 0.05

### Best Method Identification

A method is considered "best" if:
1. It has the highest mean performance (or tied for highest within statistical uncertainty)
2. It is not significantly worse (p ≥ 0.05) than the top-performing method

This approach accounts for:
- Multiple methods may be statistically tied for best
- Small differences that aren't statistically significant
- Variance across different random seeds

## Interpreting Results

### Example Output
```
Metric: llm_rating
----------------------------------------
Llama-3.2-3B-Instruct         : 3.4500 ± 0.0200
Llama-3.2-1B-Instruct         : 3.4200 ± 0.0300
combined                      : 2.8500 ± 0.1200
...

Best method(s) (statistically): Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct
```

This means:
- Llama-3.2-3B has the highest mean score
- Llama-3.2-1B is not significantly different (p ≥ 0.05)
- Both are considered "best" methods
- The combined method is significantly worse than the best methods

### Key Considerations

1. **Sample Size**: With 5 seeds, we have limited statistical power. Methods need large differences to be statistically significant.

2. **Multiple Comparisons**: When comparing many methods, some significant results may occur by chance. Consider Bonferroni correction for very conservative estimates.

3. **Effect Size**: A statistically significant difference may not be practically significant. Always consider the magnitude of differences.

4. **Metric-Specific**: Best method may differ across metrics. A method strong in coherence (NPMI) might be weaker in diversity (IRBO).

## Troubleshooting

### Missing Results
If you see warnings like "Directory not found":
- Check that all results were properly extracted
- Verify the dataset name matches the directory structure
- Ensure all 5 seeds are present for each method

### Missing Dependencies
```bash
pip install pandas scipy numpy
```

### Memory Issues
The script loads all results into memory. For very large result files:
- Process one dataset at a time
- Close other applications
- Consider increasing swap space

## Extension Ideas

1. **Add More Metrics**: Modify `metrics` list in `run_analysis()` function
2. **Different Significance Levels**: Change `alpha` parameter in statistical tests
3. **Non-parametric Tests**: Replace t-tests with Mann-Whitney U tests for non-normal data
4. **Effect Size Calculations**: Add Cohen's d for practical significance
5. **Visualization**: Create plots showing metric distributions across methods

## Notes

- **Combined = CWTM**: The "combined" method in the results refers to the Contextualized Word-Topic Model (CWTM)
- **Llama Models**: By default, uses the KL loss variant. Modify code if you want to compare CE vs KL
- **FASTopic Location**: FASTopic results are in a different directory structure than other baselines
