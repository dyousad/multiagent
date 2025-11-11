#!/bin/bash

# Test v3.1 with all short-term + iterative improvements

set -e

echo "=========================================="
echo "V3.1 TEST - All Improvements"
echo "=========================================="
echo ""
echo "Short-term improvements:"
echo "  ✓ Reasoner prompt adjusted (less 'unknown')"
echo "  ✓ Retriever Top-K increased (10→15)"
echo "  ✓ Better answer extraction (addresses, dates, numbers)"
echo ""
echo "Medium-term improvements:"
echo "  ✓ Iterative retrieval (first answer → second query)"
echo ""

# Configuration
NUM_SAMPLES=10
NUM_AGENTS=3
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo ""

# Check if cache exists
if [ ! -f "data/cache/hotpotqa_corpus_full_BAAI_bge-large-en-v1.5_embeddings.npy" ]; then
    echo "❌ Cache not found!"
    echo "Please run: python scripts/build_full_corpus_cache.py"
    exit 1
fi

echo "✓ Cache detected"
echo ""

# Create test script
cat > /tmp/test_v3_1.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "scripts"))

import json
from run_hotpotqa_experiments import run_hotpotqa_experiment

print("Running v3.1 experiment (with iterative retrieval)...")
print()

results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=10,
    num_agents=3,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    use_v3=True,
    use_iterative=True,  # Enable iterative RAG!
    output_dir=Path("results/hotpotqa_v3_1")
)

print("\n" + "="*70)
print("V3.1 RESULTS")
print("="*70)
print(f"Exact Match Accuracy: {results['aggregate']['exact_match_accuracy']:.1%}")
print(f"Average F1 Score:     {results['aggregate']['average_f1']:.3f}")
print()

# Sample-by-sample breakdown
print("="*70)
print("SAMPLE-BY-SAMPLE BREAKDOWN")
print("="*70)

for i, sample in enumerate(results['samples']):
    status = "✓ PASS" if sample['exact_match'] else "✗ FAIL"
    print(f"\nSample {i}: {status}")
    print(f"  Question: {sample['question'][:70]}...")
    print(f"  Ground Truth:  '{sample['ground_truth']}'")
    print(f"  Predicted:     '{sample['predicted_answer']}'")
    print(f"  F1 Score:      {sample['f1_score']:.3f}")

print()
print("="*70)
print("COMPARISON")
print("="*70)

# Load v3 results for comparison
versions_data = {}
versions = {
    "v0 (baseline)": "results/hotpotqa/hotpotqa_results.json",
    "v3 (enhanced, no iter)": "results/hotpotqa_v3_large/hotpotqa_results.json",
    "v3.1 (+ iterative)": "results/hotpotqa_v3_1/hotpotqa_results.json"
}

print()
print(f"{'Version':<25} {'Exact Match':<15} {'Avg F1':<10}")
print("-" * 55)

for version, path in versions.items():
    try:
        with open(path) as f:
            data = json.load(f)
            em = data['aggregate']['exact_match_accuracy']
            f1 = data['aggregate']['average_f1']
            print(f"{version:<25} {em:<15.3f} {f1:<10.3f}")
            versions_data[version] = {'em': em, 'f1': f1}
    except:
        pass

# Calculate improvement
if "v3 (enhanced, no iter)" in versions_data and "v3.1 (+ iterative)" in versions_data:
    v3_em = versions_data["v3 (enhanced, no iter)"]['em']
    v3_1_em = versions_data["v3.1 (+ iterative)"]['em']
    improvement = v3_1_em - v3_em
    print()
    print(f"Iterative improvement: {improvement:+.1%}")

print()
EOF

conda run -n vlm-anchor python /tmp/test_v3_1.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/hotpotqa_v3_1/"
echo ""
