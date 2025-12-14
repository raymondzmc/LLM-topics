================================================================================
STATISTICAL ANALYSIS OF TOPIC MODELING RESULTS - COMPLETE SETUP
================================================================================

✅ All files have been successfully extracted from the ZIP archives!
✅ Analysis scripts have been created and are ready to use!

================================================================================
QUICKEST WAY TO GET RESULTS (3 commands)
================================================================================

1. Install dependencies:
   pip install pandas scipy

2. Run the analysis:
   python run_all_analyses.py

3. View results:
   cat statistical_analysis_results/dbpedia_14_summary_table.md

That's it! You'll have statistical significance tests for all methods.

================================================================================
WHAT THIS ANALYZES
================================================================================

Methods Compared (9 total):
  • LDA, ProdLDA, Combined (CWTM), Zeroshot, ETM
  • BERTopic, FASTopic
  • Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct

For Each K Value: 25, 50, 75, 100

Across Metrics:
  • llm_rating (LLM quality assessment)
  • npmi (topic coherence)
  • inverted_rbo (topic diversity)  
  • purity (cluster purity)
  • word2vec_word_embeddings (word embedding coherence)

Using: 5 seeds per method, t-tests for significance (α=0.05)

================================================================================
FILES CREATED FOR YOU
================================================================================

Core Scripts:
  📊 statistical_analysis.py       - Main analysis engine
  🚀 run_all_analyses.py           - Easy runner (USE THIS!)
  ✅ test_statistical_analysis.py  - Validation script
  📈 visualize_results.py          - Creates plots (optional)
  🐚 run_statistical_analysis.sh   - Shell script alternative

Documentation:
  📖 ANALYSIS_QUICKSTART.md             - Quick reference guide
  📚 STATISTICAL_ANALYSIS_README.md     - Detailed documentation
  📋 SUMMARY_OF_CREATED_FILES.md        - File descriptions
  📄 README_STATISTICAL_ANALYSIS.txt    - This file

================================================================================
STEP-BY-STEP INSTRUCTIONS
================================================================================

STEP 1: Verify Setup (Optional but recommended)
   python test_statistical_analysis.py

STEP 2: Install Dependencies
   pip install pandas scipy

STEP 3: Run Analysis
   python run_all_analyses.py
   
   (This will analyze all three datasets: dbpedia_14, 20_newsgroups, stackoverflow)

STEP 4: Check Results
   Open and read the markdown files in statistical_analysis_results/:
   
   - dbpedia_14_summary_table.md  ← Start here!
   - 20_newsgroups_summary_table.md
   - stackoverflow_summary_table.md

STEP 5: (Optional) Create Visualizations
   pip install matplotlib
   python visualize_results.py

================================================================================
UNDERSTANDING THE OUTPUT
================================================================================

You'll get three types of files for each dataset:

1. {dataset}_all_metrics.csv
   → Raw data, one row per method/K/seed combination
   → Great for importing into Excel, R, or making custom plots

2. {dataset}_statistical_summary.json  
   → Complete statistics: means, stds, p-values from t-tests
   → Best methods identified for each K and metric
   → Use this for detailed statistical reporting

3. {dataset}_summary_table.md
   → Human-readable markdown tables
   → Shows best method(s) with scores
   → Rankings for all methods
   → ⭐ marks indicate statistically best methods
   → READ THIS FIRST!

================================================================================
EXAMPLE OUTPUT INTERPRETATION
================================================================================

From summary table:

| K  | llm_rating                                    |
|----|-----------------------------------------------|
| 50 | Llama-3.2-3B (3.48±0.03)                      |
|    | Llama-3.2-1B (3.42±0.04)                      |

This means:
  ✓ Llama-3.2-3B has highest mean LLM rating at K=50
  ✓ Llama-3.2-1B is not significantly different (both are "best")
  ✓ Standard deviations are small (good reproducibility)
  ✓ You can confidently use either Llama model

================================================================================
CHOOSING THE BEST METHOD
================================================================================

Different methods may excel at different metrics:

If you care most about:
  • Topic quality & interpretability → Look at llm_rating
  • Topic coherence                  → Look at npmi
  • Topic diversity                  → Look at inverted_rbo
  • Classification performance       → Look at purity

Choose the method that's best for YOUR priority metric and K value!

================================================================================
TROUBLESHOOTING
================================================================================

Problem: "Directory not found" warnings
Solution: Normal! Some methods may not have results for all datasets

Problem: "Missing dependencies" error
Solution: pip install pandas scipy

Problem: Python not found
Solution: Try "python3" instead of "python"

Problem: Want to analyze just one dataset
Solution: python statistical_analysis.py --dataset 20_newsgroups

Problem: Need more help
Solution: Read ANALYSIS_QUICKSTART.md for detailed guide

================================================================================
ALTERNATIVE RUNNING METHODS
================================================================================

Run all datasets:
  python run_all_analyses.py
  
Or use the shell script:
  chmod +x run_statistical_analysis.sh
  ./run_statistical_analysis.sh

Run individual datasets:
  python statistical_analysis.py --dataset dbpedia_14
  python statistical_analysis.py --dataset 20_newsgroups
  python statistical_analysis.py --dataset stackoverflow

Create visualizations:
  python visualize_results.py                    # All datasets
  python visualize_results.py --dataset dbpedia_14  # One dataset

================================================================================
FOR YOUR RESEARCH PAPER
================================================================================

The analysis provides:
  ✓ Mean ± Standard Deviation for each method
  ✓ P-values from pairwise t-tests
  ✓ Identification of best method(s) with statistical backing
  ✓ CSV data for creating custom publication-quality plots
  
You can report findings like:
  "Method X significantly outperformed Method Y (p < 0.05, two-sample t-test)"
  "Methods X and Y showed no significant difference (p = 0.23)"
  
Include the generated tables and plots directly in your paper!

================================================================================
WHAT'S NEXT?
================================================================================

✅ Run: python run_all_analyses.py
📖 Read: statistical_analysis_results/*_summary_table.md  
📊 Plot: python visualize_results.py (optional)
📝 Report: Use the statistics in your paper
🎯 Choose: Pick the best method for your use case

================================================================================
NEED MORE DETAILS?
================================================================================

Quick start guide:        ANALYSIS_QUICKSTART.md
Detailed documentation:   STATISTICAL_ANALYSIS_README.md
File descriptions:        SUMMARY_OF_CREATED_FILES.md
Methodology details:      STATISTICAL_ANALYSIS_README.md

================================================================================
READY TO BEGIN!
================================================================================

Run this command now:
  python run_all_analyses.py

Good luck with your analysis!
================================================================================
