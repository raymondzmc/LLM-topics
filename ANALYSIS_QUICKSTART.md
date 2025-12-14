# Quick Start Guide: Statistical Analysis of Topic Modeling Results

## 🎯 What This Does

Analyzes topic modeling results across 9 different methods and 4 K values (25, 50, 75, 100) to determine which method performs best for each configuration, using statistical significance testing.

## 📋 Prerequisites

```bash
pip install pandas scipy
```

## 🚀 Running the Analysis

### Option 1: Run Everything (Recommended)

```bash
python run_all_analyses.py
```

This will:
- Check and install dependencies
- Analyze all three datasets (dbpedia_14, 20_newsgroups, stackoverflow)
- Save results to `statistical_analysis_results/`

### Option 2: Test First, Then Run

```bash
# Test that everything is set up correctly
python test_statistical_analysis.py

# If tests pass, run the analysis
python run_all_analyses.py
```

### Option 3: Analyze Individual Datasets

```bash
# Just DBpedia
python statistical_analysis.py --dataset dbpedia_14

# Just 20 Newsgroups
python statistical_analysis.py --dataset 20_newsgroups

# Just StackOverflow
python statistical_analysis.py --dataset stackoverflow
```

## 📊 Understanding the Output

After running, you'll find these files in `statistical_analysis_results/`:

### 1. `{dataset}_all_metrics.csv`
Raw data with all metrics for every method, K value, and seed.
- Great for creating custom visualizations
- Can import into Excel, R, or Python for further analysis

### 2. `{dataset}_statistical_summary.json`
Complete statistical analysis including:
- Mean and standard deviation for each method
- p-values from pairwise t-tests
- List of best methods

### 3. `{dataset}_summary_table.md`
**Start here!** Human-readable markdown table showing:
- Best method for each K and metric
- Rankings with ⭐ marking the best performers
- Detailed statistics

Example snippet:
```markdown
## Best Methods by K and Metric

| K | llm_rating | npmi | inverted_rbo | purity |
|---|------------|------|--------------|--------|
| 25 | Llama-3.2-3B (3.45±0.02) | combined (0.12±0.01) | ... | ... |
| 50 | Llama-3.2-3B (3.48±0.03) | combined (0.14±0.02) | ... | ... |
```

## 🔍 Methods Analyzed

| Method | Description | Type |
|--------|-------------|------|
| **LDA** | Latent Dirichlet Allocation | Classical |
| **ProdLDA** | Product of Experts LDA | Neural |
| **combined (CWTM)** | Combined/Contextualized Topic Model | Neural + Contextual |
| **zeroshot** | Zero-shot CTM | Neural + Contextual |
| **ETM** | Embedded Topic Model | Neural |
| **BERTopic** | BERT-based clustering | Transformer |
| **fastopic** | Fast topic modeling | Clustering |
| **Llama-3.2-1B** | LLM-guided topic modeling (1B) | LLM |
| **Llama-3.2-3B** | LLM-guided topic modeling (3B) | LLM |

## 📈 Metrics Explained

| Metric | What It Measures | Range | Better |
|--------|------------------|-------|--------|
| **llm_rating** | LLM's quality assessment of topics | 1-5 | Higher |
| **npmi** | Normalized PMI - topic coherence | -1 to 1 | Higher |
| **inverted_rbo** | Rank-Biased Overlap - topic diversity | 0 to 1 | Higher |
| **purity** | Cluster purity for classification | 0 to 1 | Higher |
| **word2vec_word_embeddings** | Word2Vec embedding coherence | 0 to 1 | Higher |

## 📖 Interpreting Results

### What "Best Method" Means

A method is marked as "best" if:
1. ✓ It has the highest mean score, OR
2. ✓ It's statistically tied with the best (p-value ≥ 0.05)

Multiple methods can be "best" if their differences aren't statistically significant.

### Example Interpretation

```
K=50, Metric: llm_rating
----------------------------------------
Llama-3.2-3B-Instruct  : 3.4800 ± 0.0300
Llama-3.2-1B-Instruct  : 3.4200 ± 0.0400
combined               : 2.8500 ± 0.1200

Best method(s): Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct
```

**This means:**
- Both Llama models are statistically equivalent (not significantly different)
- Both outperform the combined method significantly
- If you need the best LLM rating, use either Llama model
- The 3B model has a slight edge but 1B is good enough

### When Methods Differ Across Metrics

It's common for different methods to excel at different metrics:

| Method | Best For |
|--------|----------|
| **Llama models** | Often best for LLM rating (makes sense!) |
| **Combined/CWTM** | Often strong for coherence (NPMI) |
| **BERTopic** | May excel at diversity (IRBO) |
| **Classical methods** | Sometimes competitive on purity |

**Choose based on your priorities:**
- Need interpretable topics? → Prioritize LLM rating & NPMI
- Need diverse topics? → Prioritize IRBO
- Doing classification? → Prioritize purity

## 🎓 Statistical Methodology

- **Test Used**: Two-sample independent t-test
- **Significance Level**: α = 0.05
- **Sample Size**: 5 seeds per method
- **Multiple Comparisons**: Pairwise tests between all methods

**Important:** With only 5 samples, statistical power is limited. Large differences are needed for significance.

## 🐛 Troubleshooting

### "Directory not found" warnings
- Some methods may not be available for all datasets
- Check that extraction completed successfully
- This is normal if some methods weren't run on certain datasets

### Missing metrics
- Some older results may not have all metrics
- Script will skip missing metrics and continue
- Check the CSV output to see which metrics are available

### Memory errors
- Process one dataset at a time
- Close other applications
- Each dataset loads all results into memory

### Python import errors
```bash
# Make sure you have numpy too
pip install numpy pandas scipy

# Or use the requirements from the project
pip install -r requirements.txt
pip install pandas scipy
```

## 📞 Quick Reference Commands

```bash
# Test setup
python test_statistical_analysis.py

# Run everything
python run_all_analyses.py

# Run one dataset
python statistical_analysis.py --dataset dbpedia_14

# View results
cat statistical_analysis_results/dbpedia_14_summary_table.md
```

## 📝 Next Steps

1. ✅ Run the test script to validate setup
2. ✅ Run the analysis for all datasets
3. 📊 Open the `*_summary_table.md` files to see results
4. 📈 Import `*_all_metrics.csv` for custom visualizations
5. 📄 Check JSON files for detailed statistical tests
6. 🎯 Choose best method for your specific use case

## 💡 Tips

- **Start with the markdown tables** - They're the most readable
- **Consider all metrics** - One method may not be best for everything
- **Look at standard deviations** - Large std means unstable results
- **Compare within datasets** - Don't compare across different datasets
- **Trade-offs are normal** - You may need to balance different metrics

---

For detailed documentation, see `STATISTICAL_ANALYSIS_README.md`
