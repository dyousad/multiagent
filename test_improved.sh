#!/bin/bash

# Quick test of improved HotpotQA performance
# This runs a small experiment with the improved decomposition and reasoning

set -e

echo "=========================================="
echo "Testing Improved HotpotQA Performance"
echo "=========================================="
echo ""

# Run on just 3 samples to test quickly
NUM_SAMPLES=3
NUM_AGENTS=3
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo ""

# Create a simple test script
cat > /tmp/test_improved_hotpotqa.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))

from run_hotpotqa_experiments import run_hotpotqa_experiment

print("Running improved HotpotQA experiment...")
results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=3,
    num_agents=3,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    output_dir=Path("results/hotpotqa_improved")
)

print("\n" + "="*70)
print("IMPROVED RESULTS SUMMARY")
print("="*70)
print(f"Exact Match Accuracy: {results['aggregate']['exact_match_accuracy']:.3f}")
print(f"Average F1 Score:     {results['aggregate']['average_f1']:.3f}")
print("")
print("Sample-by-sample breakdown:")
for i, sample in enumerate(results['samples']):
    print(f"\nSample {i}:")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Ground Truth: {sample['ground_truth']}")
    print(f"  Predicted:    {sample['predicted_answer'][:60]}...")
    print(f"  Exact Match:  {sample['exact_match']}")
    print(f"  F1 Score:     {sample['f1_score']:.3f}")
EOF

python /tmp/test_improved_hotpotqa.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/hotpotqa_improved/"
echo ""
