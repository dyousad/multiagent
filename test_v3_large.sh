#!/bin/bash

# Large scale test with 50 samples

set -e

echo "=========================================="
echo "V3 LARGE SCALE TEST - 50 Samples"
echo "=========================================="
echo ""
echo "This test will evaluate v3 enhancements on 50 samples"
echo "to assess performance and stability."
echo ""
echo "Key improvements:"
echo "  ✓ Reasoner v3 (stronger inference, better extraction)"
echo "  ✓ Retriever v3 (top-k=10, better ranking)"
echo "  ✓ Multi-stage reranking with diversity"
echo "  ✓ Enhanced yes/no detection"
echo ""

# Configuration
NUM_SAMPLES=100
NUM_AGENTS=3
MODEL="deepseek-ai/DeepSeek-V3"

echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Agents: $NUM_AGENTS"
echo "  Model: $MODEL"
echo ""
echo "⏱️  Estimated time: ~10-15 minutes"
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
cat > /tmp/test_v3_large.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "scripts"))

import json
from run_hotpotqa_experiments import run_hotpotqa_experiment

print("Running v3 experiment on 50 samples...")
print("This will take approximately 10-15 minutes.\n")

results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=100,
    num_agents=3,
    model_identifier="deepseek-ai/DeepSeek-V3",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    use_v3=True,
    output_dir=Path("results/hotpotqa_v3_large")
)

print("\n" + "="*70)
print("V3 LARGE SCALE RESULTS (50 Samples)")
print("="*70)
print(f"Exact Match Accuracy: {results['aggregate']['exact_match_accuracy']:.1%}")
print(f"Average F1 Score:     {results['aggregate']['average_f1']:.3f}")
print()

# Analyze by question type
samples = results['samples']
correct = sum(1 for s in samples if s['exact_match'])
total = len(samples)

print(f"Correct: {correct}/{total}")
print()

# Group failures
print("="*70)
print("FAILURE ANALYSIS")
print("="*70)
print()

failures = [s for s in samples if not s['exact_match']]
print(f"Total failures: {len(failures)}")
print()

if failures:
    print("Failed questions (showing first 10):")
    for i, sample in enumerate(failures[:10], 1):
        print(f"\n{i}. Question: {sample['question'][:70]}...")
        print(f"   Ground Truth: '{sample['ground_truth']}'")
        print(f"   Predicted:    '{sample['predicted_answer']}'")
        print(f"   F1 Score:     {sample['f1_score']:.3f}")

print()
print("="*70)
print("SUCCESS ANALYSIS")
print("="*70)
print()

successes = [s for s in samples if s['exact_match']]
print(f"Total successes: {len(successes)}")

# Categorize by question type
yes_no_questions = [s for s in successes if s['ground_truth'].lower() in ['yes', 'no']]
entity_questions = [s for s in successes if s['ground_truth'].lower() not in ['yes', 'no']]

print(f"  Yes/No questions: {len(yes_no_questions)}")
print(f"  Entity questions: {len(entity_questions)}")

print()
print("="*70)
print("DETAILED METRICS")
print("="*70)
print()

# Calculate metrics by question type
all_yes_no = [s for s in samples if s['ground_truth'].lower() in ['yes', 'no']]
all_entity = [s for s in samples if s['ground_truth'].lower() not in ['yes', 'no']]

if all_yes_no:
    yes_no_acc = sum(1 for s in all_yes_no if s['exact_match']) / len(all_yes_no)
    print(f"Yes/No Questions Accuracy: {yes_no_acc:.1%} ({sum(1 for s in all_yes_no if s['exact_match'])}/{len(all_yes_no)})")

if all_entity:
    entity_acc = sum(1 for s in all_entity if s['exact_match']) / len(all_entity)
    print(f"Entity Questions Accuracy: {entity_acc:.1%} ({sum(1 for s in all_entity if s['exact_match'])}/{len(all_entity)})")

print()

# Calculate average F1 for different groups
if all_yes_no:
    avg_f1_yes_no = sum(s['f1_score'] for s in all_yes_no) / len(all_yes_no)
    print(f"Average F1 (Yes/No):  {avg_f1_yes_no:.3f}")

if all_entity:
    avg_f1_entity = sum(s['f1_score'] for s in all_entity) / len(all_entity)
    print(f"Average F1 (Entity):  {avg_f1_entity:.3f}")

print()
print("="*70)
print("COMPARISON WITH BASELINE")
print("="*70)
print()

# Load baseline results if available
baseline_path = "results/hotpotqa/hotpotqa_results.json"
try:
    with open(baseline_path) as f:
        baseline = json.load(f)
        baseline_em = baseline['aggregate']['exact_match_accuracy']
        baseline_f1 = baseline['aggregate']['average_f1']

        print(f"{'Version':<20} {'Exact Match':<15} {'Avg F1':<10}")
        print("-" * 50)
        print(f"{'v0 (baseline)':<20} {baseline_em:<15.3f} {baseline_f1:<10.3f}")
        print(f"{'v3 (enhanced)':<20} {results['aggregate']['exact_match_accuracy']:<15.3f} {results['aggregate']['average_f1']:<10.3f}")

        improvement = results['aggregate']['exact_match_accuracy'] - baseline_em
        print()
        print(f"Improvement: {improvement:+.1%}")
except:
    print("Baseline results not found for comparison")

print()
print("="*70)
print("RESULTS SAVED")
print("="*70)
print()
print(f"Full results: results/hotpotqa_v3_large/hotpotqa_results.json")
print(f"Individual samples: results/hotpotqa_v3_large/task_*.json")
print()
EOF

python /tmp/test_v3_large.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/hotpotqa_v3_large/"
echo ""
