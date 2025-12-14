# Summary of Created Files for Statistical Analysis

## 📦 What Was Created

I've created a complete statistical analysis toolkit for your topic modeling results. Here's what each file does:

### Core Analysis Scripts

1. **`statistical_analysis.py`** (Main analysis script)
   - Loads results from all methods (LDA, ProdLDA, Combined/CWTM, Zeroshot, ETM, BERTopic, FASTopic, Llama-3.2-1B, Llama-3.2-3B)
   - Performs statistical significance testing (t-tests) between all method pairs
   - Identifies the best method(s) for each K value and metric
   - Outputs: CSV (raw data), JSON (statistics), and Markdown (summary tables)

2. **`run_all_analyses.py`** (Convenience runner)
   - Automatically installs dependencies (pandas, scipy)
   - Runs analysis for all three datasets
   - Handles errors gracefully
   - **USE THIS TO START!**

3. **`test_statistical_analysis.py`** (Validation script)
   - Tests file structure
   - Validates result files can be loaded
   - Checks for FASTopic and Llama model results
   - Counts available methods per dataset
   - Run this first to ensure everything is set up correctly

4. **`visualize_results.py`** (Visualization script)
   - Creates comparison bar plots for each metric
   - Generates heatmaps showing method×K performance
   - Makes radar plots comparing top methods across metrics
   - Requires: `pip install matplotlib`

### Shell Scripts

5. **`run_statistical_analysis.sh`** (Bash runner)
   - Alternative to Python runner for Unix/Mac users
   - Does the same as `run_all_analyses.py`
   - Make executable: `chmod +x run_statistical_analysis.sh`

### Documentation

6. **`ANALYSIS_QUICKSTART.md`** (Quick reference guide)
   - Step-by-step instructions
   - Command examples
   - Output interpretation
   - **START HERE!**

7. **`STATISTICAL_ANALYSIS_README.md`** (Detailed documentation)
   - Complete methodology explanation
   - Statistical testing details
   - Extension ideas
   - Troubleshooting guide

8. **`SUMMARY_OF_CREATED_FILES.md`** (This file)
   - Overview of all created files
   - Next steps

## 🚀 Quick Start (3 Steps)

```bash
# Step 1: Test the setup
python test_statistical_analysis.py

# Step 2: Run the analysis
python run_all_analyses.py

# Step 3: View results
cat statistical_analysis_results/20_newsgroups_summary_table.md
```

That's it! You'll have complete statistical analysis results.

## 📊 What You'll Get

After running the analysis, you'll have:

```
statistical_analysis_results/
├── 20_newsgroups_all_metrics.csv          # Raw data (import to Excel/R/Python)
├── 20_newsgroups_statistical_summary.json  # Complete statistics + p-values
├── 20_newsgroups_summary_table.md         # Human-readable results ⭐ READ THIS
├── dbpedia_14_all_metrics.csv
├── dbpedia_14_statistical_summary.json
├── dbpedia_14_summary_table.md
├── stackoverflow_all_metrics.csv
├── stackoverflow_statistical_summary.json
├── stackoverflow_summary_table.md
└── plots/                                  # (if you run visualization)
    ├── 20_newsgroups_llm_rating_comparison.png
    ├── 20_newsgroups_npmi_heatmap.png
    ├── 20_newsgroups_radar_K50.png
    └── ...
```

## 🎯 Understanding "Best Method"

The script identifies the statistically best method(s) using:

1. **Pairwise t-tests** between all method pairs
2. **Significance level** α = 0.05
3. **Multiple methods can be "best"** if not significantly different

Example output:
```
K=50, Metric: llm_rating
-----------------------------------------
Llama-3.2-3B-Instruct  : 3.4800 ± 0.0300 ⭐
Llama-3.2-1B-Instruct  : 3.4200 ± 0.0400 ⭐
combined               : 2.8500 ± 0.1200
```

Both Llama models are marked "best" because they're not significantly different (p ≥ 0.05).

## 📈 Metrics Analyzed

For each method and K value (25, 50, 75, 100):

| Metric | Measures | Higher is Better? |
|--------|----------|------------------|
| **llm_rating** | LLM quality assessment | ✅ Yes |
| **npmi** | Topic coherence (PMI) | ✅ Yes |
| **inverted_rbo** | Topic diversity | ✅ Yes |
| **purity** | Cluster purity | ✅ Yes |

## 🔧 Dependencies

### Required (for analysis):
```bash
pip install pandas scipy
```

### Optional (for visualization):
```bash
pip install matplotlib
```

### Already installed (from your requirements.txt):
```bash
numpy  # Already have via torch/other packages
```

## 📝 Methods Compared

| ID | Method Name | Notes |
|----|-------------|-------|
| 1  | lda | Latent Dirichlet Allocation |
| 2  | prodlda | Product of Experts LDA |
| 3  | combined | **This is CWTM** (Combined CTM) |
| 4  | zeroshot | Zero-shot CTM |
| 5  | etm | Embedded Topic Model |
| 6  | bertopic | BERT-based topic model |
| 7  | fastopic | Fast topic modeling |
| 8  | Llama-3.2-1B-Instruct | Uses KL loss variant |
| 9  | Llama-3.2-3B-Instruct | Uses KL loss variant |

**Note:** "combined" = CWTM (Contextualized Word-Topic Model)

## 🎨 Optional: Creating Visualizations

After running the analysis:

```bash
# Install matplotlib if not already installed
pip install matplotlib

# Create all plots
python visualize_results.py

# Or just one dataset
python visualize_results.py --dataset 20_newsgroups
```

This creates:
- Bar charts comparing methods
- Heatmaps showing performance across K values
- Radar plots comparing top methods on all metrics

## 🐛 Common Issues

### Issue: "Directory not found" warnings
**Solution:** Some methods may not have results for all datasets. This is normal.

### Issue: "Missing dependencies"
**Solution:** Run `pip install pandas scipy`

### Issue: Shell script won't run
**Solution:** 
```bash
chmod +x run_statistical_analysis.sh
./run_statistical_analysis.sh
```
Or just use the Python version: `python run_all_analyses.py`

### Issue: Need to change significance level
**Solution:** Edit `statistical_analysis.py`, line with `alpha=0.05`, change to desired value

## 📚 File Relationships

```
┌─────────────────────────────────┐
│   ANALYSIS_QUICKSTART.md        │  ← START HERE (how to use)
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│   test_statistical_analysis.py  │  ← Run first (validation)
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│   run_all_analyses.py           │  ← Main runner
│   (or run_statistical_analysis.sh)
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│   statistical_analysis.py       │  ← Core analysis logic
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│   statistical_analysis_results/ │  ← Output directory
│     ├── *.csv (raw data)
│     ├── *.json (statistics)
│     └── *.md (tables) ← READ THIS
└─────────────────────────────────┘
                ↓ (optional)
┌─────────────────────────────────┐
│   visualize_results.py          │  ← Make plots
└─────────────────────────────────┘
                ↓
┌─────────────────────────────────┐
│   plots/ (charts & heatmaps)    │
└─────────────────────────────────┘
```

## 🎓 For Your Paper

The analysis provides publication-ready statistics:

1. **Mean ± Std** for each method
2. **p-values** for pairwise comparisons
3. **Best method identification** with statistical backing
4. **CSV data** for creating custom plots in R/Python
5. **Multiple testing contexts** (different K, different metrics)

You can report:
- "Method X significantly outperformed Method Y (p < 0.05, t-test)"
- "Methods X and Y showed no significant difference (p = 0.23)"
- Include the mean ± std values in tables
- Use the heatmaps/plots in your paper

## 💡 Next Steps

1. ✅ **Run the test:** `python test_statistical_analysis.py`
2. ✅ **Run the analysis:** `python run_all_analyses.py`
3. 📖 **Read the results:** Open `statistical_analysis_results/*_summary_table.md`
4. 📊 **Create plots (optional):** `python visualize_results.py`
5. 📝 **Use in your paper:** Import CSV data or use the markdown tables
6. 🎯 **Choose your method:** Based on which metrics matter most to you

## 📞 Quick Reference

| I want to... | Command |
|--------------|---------|
| Test everything works | `python test_statistical_analysis.py` |
| Run full analysis | `python run_all_analyses.py` |
| Analyze just one dataset | `python statistical_analysis.py --dataset 20_newsgroups` |
| Make plots | `python visualize_results.py` |
| See results quickly | `cat statistical_analysis_results/20_newsgroups_summary_table.md` |
| Get help | Read `ANALYSIS_QUICKSTART.md` |
| Understand methodology | Read `STATISTICAL_ANALYSIS_README.md` |

---

**Ready to go!** Start with: `python run_all_analyses.py`

