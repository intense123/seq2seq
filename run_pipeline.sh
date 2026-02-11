#!/bin/bash

# Complete pipeline for Text-to-Code Generation project
# This script runs all steps: data preparation, training, evaluation, and visualization

set -e  # Exit on error

echo "=========================================="
echo "Text-to-Code Generation Pipeline"
echo "=========================================="

# Step 1: Prepare data
echo ""
echo "Step 1: Preparing dataset..."
echo "=========================================="
python -c "from data.preprocess import prepare_data; prepare_data()"

# Step 2: Train all models
echo ""
echo "Step 2: Training models..."
echo "=========================================="
python train.py --model all --num_epochs 20 --batch_size 32

# Step 3: Evaluate all models
echo ""
echo "Step 3: Evaluating models..."
echo "=========================================="
python evaluate.py --model all --batch_size 32

# Step 4: Visualize attention
echo ""
echo "Step 4: Visualizing attention..."
echo "=========================================="
python visualize_attention.py --num_examples 5

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Results: ./results/"
echo "  - Plots: ./results/plots/"
echo ""
echo "Next steps:"
echo "  1. Review training curves in ./results/plots/"
echo "  2. Check evaluation metrics in ./results/evaluation_results.json"
echo "  3. Examine attention heatmaps in ./results/plots/"
echo "  4. Read attention analysis in ./results/attention_analysis.txt"
echo "  5. Review example predictions in ./results/*_examples.txt"
