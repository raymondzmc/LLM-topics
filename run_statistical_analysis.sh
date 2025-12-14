#!/bin/bash

# Install dependencies if not present
echo "Installing dependencies..."
pip install pandas scipy --quiet

# Create output directory
mkdir -p statistical_analysis_results

# Run analysis for each dataset
echo ""
echo "============================================"
echo "Running statistical analysis..."
echo "============================================"
echo ""

for dataset in "dbpedia_14" "20_newsgroups" "stackoverflow"
do
    echo "Analyzing $dataset..."
    python statistical_analysis.py --dataset $dataset --output_dir statistical_analysis_results
    echo ""
done

echo "============================================"
echo "Analysis complete!"
echo "Results saved in: statistical_analysis_results/"
echo "============================================"
