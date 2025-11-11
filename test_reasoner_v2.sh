#!/bin/bash

# Test improved Reasoner v2 with better inference capabilities

set -e

echo "=========================================="
echo "Testing Reasoner v2 - Enhanced Inference"
echo "=========================================="
echo ""

# Run on the same 3 samples to compare
NUM_SAMPLES=3
NUM_AGENTS=3
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo "  Changes: Enhanced reasoning with contextual inference"
echo ""

# Create test script
cat > /tmp/test_reasoner_v2.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))

from run_hotpotqa_experiments import run_hotpotqa_experiment

print("Running Reasoner v2 experiment with enhanced inference...")
results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=3,
    num_agents=3,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    output_dir=Path("results/hotpotqa_v2")
)

print("\n" + "="*70)
print("REASONER V2 RESULTS")
print("="*70)
print(f"Exact Match Accuracy: {results['aggregate']['exact_match_accuracy']:.3f}")
print(f"Average F1 Score:     {results['aggregate']['average_f1']:.3f}")
print("")

# Compare with previous versions
import json

# Load v1 results
try:
    with open('results/hotpotqa_improved/hotpotqa_results.json') as f:
        v1 = json.load(f)
    print("Comparison with v1 (strict evidence-only):")
    print(f"  v1 Exact Match: {v1['aggregate']['exact_match_accuracy']:.3f}")
    print(f"  v2 Exact Match: {results['aggregate']['exact_match_accuracy']:.3f}")
    print(f"  Improvement: {results['aggregate']['exact_match_accuracy'] - v1['aggregate']['exact_match_accuracy']:+.3f}")
except:
    pass

print("")
print("Sample-by-sample breakdown:")
for i, sample in enumerate(results['samples']):
    print(f"\nSample {i}:")
    print(f"  Question: {sample['question'][:60]}...")
    print(f"  Ground Truth:  '{sample['ground_truth']}'")
    print(f"  Predicted:     '{sample['predicted_answer']}'")
    print(f"  Exact Match:   {'✓ PASS' if sample['exact_match'] else '✗ FAIL'}")
    print(f"  F1 Score:      {sample['f1_score']:.3f}")
EOF

python /tmp/test_reasoner_v2.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/hotpotqa_v2/"
echo ""
