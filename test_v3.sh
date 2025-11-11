#!/bin/bash

# Test v3 improvements with enhanced Reasoner and Retriever

set -e

echo "=========================================="
echo "V3 TEST - Enhanced Reasoning & Retrieval"
echo "=========================================="
echo ""
echo "Key improvements:"
echo "  ✓ Reasoner v3 (stronger inference, better extraction)"
echo "  ✓ Retriever v3 (top-k=10, better ranking)"
echo "  ✓ Multi-stage reranking with diversity"
echo "  ✓ Enhanced yes/no detection"
echo ""

# Configuration
NUM_SAMPLES=5
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
cat > /tmp/test_v3.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "scripts"))

import json
from run_hotpotqa_experiments import run_hotpotqa_experiment

print("Running v3 experiment...")
print()

results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=5,
    num_agents=3,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    use_v3=True,  # Enable v3 enhancements
    output_dir=Path("results/hotpotqa_v3")
)

print("\n" + "="*70)
print("V3 RESULTS")
print("="*70)
print(f"Exact Match Accuracy: {results['aggregate']['exact_match_accuracy']:.3f}")
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

    # Show evidence quality
    if 'evidence_paths' in sample and sample['evidence_paths']:
        print(f"  Evidence retrieved: {len(sample['evidence_paths'])} passages")

print()
print("="*70)
print("COMPARISON WITH PREVIOUS VERSIONS")
print("="*70)

# Load previous results for comparison
versions = {
    "v0 (baseline)": "results/hotpotqa/hotpotqa_results.json",
    "v3 (enhanced)": "results/hotpotqa_v3/hotpotqa_results.json"
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
    except:
        pass

print()
EOF

python /tmp/test_v3.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/hotpotqa_v3/"
echo ""
