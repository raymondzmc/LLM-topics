# Statistical Analysis with K-Averaged Results

## Purpose

When your main results table reports **performance averaged across K values** (e.g., mean over K ∈ {25, 50, 75, 100}), you need appropriate statistical tests for those averaged values. This script provides that functionality.

## The Problem

Standard approach tests each K separately:
- "Method A is best at K=25" (p < 0.05)
- "Method B is best at K=50" (p < 0.05)
- etc.

But your main table shows:
- "Method A: 0.85 (averaged over all K)"
- "Method B: 0.82 (averaged over all K)"

**Question:** Is Method A significantly better than Method B *when averaged across K*?

## The Solution

This script implements **two strategies** for computing significance on K-averaged results:

### Strategy 1: "averaged" (Recommended for Main Tables)

**Approach:**
1. For each method and each seed, first average performance across all K values
2. This gives you 5 values per method (one per seed)
3. Perform t-test on these 5 averaged values

**When to use:** When your main table shows averages across K

**Interpretation:** "Method A's average performance (across K) is significantly better than Method B's average performance"

**Sample size:** n = 5 per method

### Strategy 2: "expanded" (More Conservative)

**Approach:**
1. Treat each K×seed combination as independent
2. This gives you up to 20 values per method (5 seeds × 4 K values)
3. Perform t-test on all 20 values

**When to use:** For more conservative estimates or sensitivity analysis

**Interpretation:** "Method A is significantly better than Method B across all configurations"

**Sample size:** n = 20 per method

## Usage

### Quick Start

```bash
# Run all datasets with recommended strategy
python run_averaged_k_analysis.py
```

### Individual Datasets

```bash
# DBPedia with averaged strategy (recommended)
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy averaged

# 20 Newsgroups
python statistical_analysis_averaged_k.py --dataset 20_newsgroups --strategy averaged

# StackOverflow
python statistical_analysis_averaged_k.py --dataset stackoverflow --strategy averaged
```

### Try Both Strategies

```bash
# Averaged approach (n=5)
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy averaged

# Expanded approach (n=20, more conservative)
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy expanded
```

## Output Files

For each dataset, you get:

1. **`{dataset}_averaged_K_metrics_{strategy}.csv`**
   - Raw data for each method
   - Averaged strategy: 5 rows per method (one per seed)
   - Expanded strategy: up to 20 rows per method (seed × K combinations)

2. **`{dataset}_averaged_K_summary_{strategy}.json`**
   - Complete statistics: means, stds, p-values
   - Best methods identified
   - All pairwise comparisons

3. **`{dataset}_averaged_K_summary_{strategy}.md`** ⭐
   - Human-readable summary table
   - Best methods for each metric
   - Rankings with significance markers

## Comparison: K-Specific vs K-Averaged

### Per-K Analysis (original script)
```
K=25: Method A (0.90 ± 0.02) > Method B (0.85 ± 0.03) *
K=50: Method B (0.88 ± 0.02) > Method A (0.86 ± 0.03) *
K=75: Method A (0.87 ± 0.02) ≈ Method B (0.86 ± 0.02)
K=100: Method A (0.85 ± 0.03) > Method B (0.82 ± 0.02) *
```

### K-Averaged Analysis (this script)
```
Averaged: Method A (0.87 ± 0.02) > Method B (0.85 ± 0.02) *
```

## Statistical Interpretation

### For "averaged" strategy (n=5):

```python
# Example results:
Method A: 0.870 ± 0.020 (n=5)
Method B: 0.850 ± 0.025 (n=5)
p-value: 0.032

Interpretation:
"When averaged across K ∈ {25, 50, 75, 100}, Method A (0.870 ± 0.020) 
significantly outperformed Method B (0.850 ± 0.025) with p = 0.032 
(two-sample t-test, n = 5 seeds)."
```

### For "expanded" strategy (n=20):

```python
# Example results:
Method A: 0.870 ± 0.035 (n=20)
Method B: 0.850 ± 0.040 (n=20)
p-value: 0.048

Interpretation:
"Method A (0.870 ± 0.035) significantly outperformed Method B 
(0.850 ± 0.040) across all K values and seeds (p = 0.048, 
two-sample t-test, n = 20 configurations)."
```

## Which Strategy to Use?

### Use "averaged" when:
- ✅ Your main table shows K-averaged results
- ✅ You want to test the averaged performance claim
- ✅ You care about overall performance across K
- ✅ Standard approach for most papers

### Use "expanded" when:
- ✅ You want more conservative estimates
- ✅ Doing sensitivity analysis
- ✅ Want to show robustness across configurations
- ✅ Reviewers ask for it

### Report both when:
- ✅ Results differ between strategies (interesting finding!)
- ✅ You want to be thorough
- ✅ Space allows

## Example Paper Text

### In Methods Section:

```latex
To assess statistical significance of averaged performance, we first computed the 
mean performance across K ∈ \{25, 50, 75, 100\} for each random seed, yielding 
5 averaged values per method. We then performed pairwise two-sample t-tests 
(Welch's t-test, α = 0.05) between all method pairs to identify statistically 
best methods.
```

### In Results Section:

```latex
When averaged across all K values, our method achieved a mean LLM rating of 
2.94 ± 0.02 (averaged over 5 seeds), significantly outperforming all baseline 
methods (p < 0.05 for all comparisons). The strongest baseline, Combined CTM, 
achieved 2.65 ± 0.07, representing a significant gap (t = 8.2, p < 0.001).
```

### In Table Caption:

```latex
\caption{Performance averaged across K ∈ \{25, 50, 75, 100\}. Values show 
mean ± standard deviation over 5 random seeds. \textbf{Bold} indicates 
statistically best methods determined by pairwise t-tests on K-averaged 
results (α = 0.05).}
```

## Theoretical Justification

### Why this is valid:

1. **Repeated Measures:** Each seed provides a K-averaged observation
2. **Independence:** Different seeds are independent samples
3. **Central Limit Theorem:** With 5 seeds, t-test is appropriate
4. **Welch's t-test:** Doesn't assume equal variances between methods

### Assumptions:

- ✅ Random seeds are independent
- ✅ Averaging across K is meaningful for your application
- ✅ All K values have roughly equal importance
- ✅ Sample size (n=5) is standard in ML/NLP

## Comparison with Alternatives

| Method | n per method | Conservative? | Complexity | Use when |
|--------|--------------|---------------|------------|----------|
| **Per-K tests** | 5 per K | Medium | Simple | Reporting K-specific results |
| **Averaged** | 5 total | Medium | Simple | **Main table with K-averaged** |
| **Expanded** | 20 total | More | Simple | Sensitivity analysis |
| **Repeated measures ANOVA** | 20 total | Less | Complex | Detailed analysis |
| **Mixed-effects** | 20 total | Medium | Very complex | Publication in stats journals |

## FAQ

**Q: Which script should I use for my main table?**  
A: Use `statistical_analysis_averaged_k.py` with `--strategy averaged`

**Q: The p-values are different from per-K analysis. Is this wrong?**  
A: No! You're testing different hypotheses. Per-K tests each K separately; this tests the average.

**Q: Should sample size be n=5 or n=20 in my paper?**  
A: For "averaged" strategy, report n=5. For "expanded", report n=20.

**Q: Can multiple methods be "best"?**  
A: Yes! Methods are marked "best" if not significantly different from the top performer.

**Q: What if results differ between "averaged" and "expanded"?**  
A: This is informative! It means performance varies across K. Discuss in your paper.

## Example Commands

```bash
# Generate K-averaged analysis for all datasets (recommended)
python run_averaged_k_analysis.py

# Just one dataset
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy averaged

# Compare both strategies
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy averaged
python statistical_analysis_averaged_k.py --dataset dbpedia_14 --strategy expanded

# View results
cat statistical_analysis_results/dbpedia_14_averaged_K_summary_averaged.md
```

## References

For statistical methodology:

```bibtex
@article{welch1947generalization,
  title={The generalization of 'Student's' problem when several different population variances are involved},
  author={Welch, Bernard L},
  journal={Biometrika},
  volume={34},
  number={1/2},
  pages={28--35},
  year={1947}
}

@inproceedings{dror2018hitchhiker,
  title={The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing},
  author={Dror, Rotem and Baumer, Gili and Shlomov, Segev and Reichart, Roi},
  booktitle={Proceedings of ACL},
  year={2018}
}
```

---

**Ready to use!** Run `python run_averaged_k_analysis.py` to get started.

