#!/bin/bash

# Complete workflow from scratch for HotPotQA Multi-Agent System
# This script will build cache from scratch and run the full experiment

set -e  # Exit on error

echo "=========================================="
echo "HotpotQA Complete Workflow from Scratch"
echo "=========================================="
echo ""

# Step 1: Check environment
echo "Step 1: Checking conda environment..."
echo "----------------------------------------"
conda activate vlm-anchor 2>/dev/null || {
    echo "❌ Error: conda environment 'vlm-anchor' not found"
    echo "Please create the environment first or use a different environment"
    exit 1
}
echo "✓ Using conda environment: vlm-anchor"
echo ""

# Step 2: Build corpus cache from scratch
echo "Step 2: Building corpus cache from scratch..."
echo "----------------------------------------"
echo "This will take several minutes..."
python scripts/build_full_corpus_cache.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Cache building failed!"
    exit 1
fi
echo ""

# Step 3: Run the complete HotPotQA workflow
echo "Step 3: Running complete HotPotQA workflow..."
echo "----------------------------------------"
./run_hotpotqa.sh "$@"

echo ""
echo "=========================================="
echo "Complete Workflow Finished!"
echo "=========================================="
echo ""
echo "This workflow included:"
echo "  1. Building full corpus cache from scratch"
echo "  2. Running integration tests"
echo "  3. Running HotPotQA experiments"
echo "  4. Generating visualizations"
echo "  5. Showing results summary"
echo ""