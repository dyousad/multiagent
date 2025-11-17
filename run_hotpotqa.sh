#!/bin/bash

# HotpotQA Multi-Agent Evaluation - Quick Start Script

set -e  # Exit on error

echo "=========================================="
echo "HotpotQA Multi-Agent Evaluation"
echo "=========================================="
echo ""

# Check if data file exists
DATA_FILE="data/hotpot_dev_fullwiki_v1.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: HotpotQA data file not found at $DATA_FILE"
    echo ""
    echo "Please download the HotpotQA dataset:"
    echo "  https://hotpotqa.github.io/"
    echo ""
    echo "Expected file location: $DATA_FILE"
    exit 1
fi

echo "✓ Data file found: $DATA_FILE"
echo ""

# Parse command line arguments
NUM_SAMPLES=10
NUM_AGENTS=3
MODEL="Qwen/Qwen2.5-7B-Instruct"
USE_DECOMPOSER=True
USE_DYNAMIC=True

while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --no-decomposer)
            USE_DECOMPOSER=false
            shift
            ;;
        --no-dynamic)
            USE_DYNAMIC=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --samples N          Number of samples to evaluate (default: 10)"
            echo "  --agents N           Number of agents (default: 3)"
            echo "  --model MODEL        Model identifier (default: deepseek-ai/DeepSeek-V3)"
            echo "  --no-decomposer      Disable question decomposer"
            echo "  --no-dynamic         Disable dynamic credit allocation"
            echo "  --help               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo "  Question Decomposer: $USE_DECOMPOSER"
echo "  Dynamic Credit: $USE_DYNAMIC"
echo ""

# Step 1: Run tests
echo "Step 1: Running integration tests..."
echo "----------------------------------------"
python3 scripts/test_hotpotqa_integration.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Please fix errors before running experiments."
    exit 1
fi
echo ""

# Step 2: Run experiments
echo "Step 2: Running HotpotQA experiments..."
echo "----------------------------------------"

# Create a temporary Python script to run with custom parameters
cat > /tmp/run_hotpotqa_custom.py <<EOF
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
from run_hotpotqa_experiments import run_hotpotqa_experiment

results = run_hotpotqa_experiment(
    data_path="$DATA_FILE",
    max_samples=$NUM_SAMPLES,
    num_agents=$NUM_AGENTS,
    model_identifier="$MODEL",
    use_decomposer=$USE_DECOMPOSER,
    use_dynamic_credit=$USE_DYNAMIC,
    output_dir=Path("results/hotpotqa")
)
EOF

python3 /tmp/run_hotpotqa_custom.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Experiments failed!"
    exit 1
fi
echo ""

# Step 3: Generate visualizations
echo "Step 3: Generating visualizations..."
echo "----------------------------------------"
python3 scripts/plot_hotpotqa_results.py \
    --results results/hotpotqa/hotpotqa_results.json \
    --output_dir results/hotpotqa/plots

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Visualization generation failed, but results are saved."
else
    echo ""
    echo "✓ Visualizations saved to results/hotpotqa/plots/"
fi
echo ""

# Step 4: Show summary
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  JSON results: results/hotpotqa/hotpotqa_results.json"
echo "  Plots:        results/hotpotqa/plots/"
echo ""
echo "Quick view of results:"
if [ -f "results/hotpotqa/hotpotqa_results.json" ]; then
    python3 -c "
import json
with open('results/hotpotqa/hotpotqa_results.json') as f:
    data = json.load(f)
    agg = data.get('aggregate', {})
    print(f\"  Exact Match Accuracy: {agg.get('exact_match_accuracy', 0):.3f}\")
    print(f\"  Average F1 Score:     {agg.get('average_f1', 0):.3f}\")
    print(f\"  Static Entropy:       {agg.get('average_static_entropy', 0):.3f}\")
    if 'average_dynamic_entropy' in agg:
        print(f\"  Dynamic Entropy:      {agg.get('average_dynamic_entropy', 0):.3f}\")
        print(f\"  Entropy Improvement:  {agg.get('average_dynamic_entropy', 0) - agg.get('average_static_entropy', 0):+.3f}\")
"
fi
echo ""
echo "For more details, see:"
echo "  - HOTPOTQA_GUIDE.md for usage guide"
echo "  - HOTPOTQA_SUMMARY.md for implementation summary"
echo ""
