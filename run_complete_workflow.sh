#!/bin/bash

# Complete workflow from scratch for HotPotQA Multi-Agent System
# This script will build cache from scratch and run the full experiment

echo "=========================================="
echo "HotpotQA Complete Workflow from Scratch"
echo "=========================================="
echo ""

# Step 1: Setup Python environment
echo "Step 1: Setting up Python environment..."
echo "----------------------------------------"

# Check if conda is available and vlm-anchor environment exists
if command -v conda >/dev/null 2>&1; then
    echo "Conda detected, checking for vlm-anchor environment..."
    if conda env list | grep -q "vlm-anchor"; then
        echo "✓ Found conda environment 'vlm-anchor', activating..."
        conda activate vlm-anchor
        echo "✓ Using conda environment: vlm-anchor"
    else
        echo "⚠️  Conda environment 'vlm-anchor' not found"
        echo "Creating Python virtual environment instead..."
        python -m venv vlm-anchor-venv
        source vlm-anchor-venv/bin/activate
        echo "✓ Created and activated virtual environment: vlm-anchor-venv"

        # Install requirements if file exists
        if [ -f "requirements.txt" ]; then
            echo "Installing dependencies from requirements.txt..."
            pip install -r requirements.txt
        else
            echo "Installing basic dependencies..."
            pip install numpy torch transformers sentence-transformers faiss-cpu openai
        fi
    fi
else
    echo "Conda not found, using Python virtual environment..."
    if [ ! -d "vlm-anchor-venv" ]; then
        echo "Creating Python virtual environment..."
        python -m venv vlm-anchor-venv
        source vlm-anchor-venv/bin/activate
        echo "✓ Created and activated virtual environment: vlm-anchor-venv"

        # Install requirements if file exists
        if [ -f "requirements.txt" ]; then
            echo "Installing dependencies from requirements.txt..."
            pip install -r requirements.txt
        else
            echo "Installing basic dependencies..."
            pip install numpy torch transformers sentence-transformers faiss-cpu openai
        fi
    else
        echo "✓ Found existing virtual environment, activating..."
        source vlm-anchor-venv/bin/activate
        echo "✓ Using virtual environment: vlm-anchor-venv"
    fi
fi
echo ""

# Enable exit on error after environment setup
set -e

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
echo "  1. Setting up Python environment (conda or venv)"
echo "  2. Building full corpus cache from scratch"
echo "  3. Running integration tests"
echo "  4. Running HotPotQA experiments"
echo "  5. Generating visualizations"
echo "  6. Showing results summary"
echo ""