#!/bin/bash

# Quick debug test for iterative RAG

set -e

echo "Testing iterative RAG agent detection..."

cat > /tmp/test_iter_debug.py <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "scripts"))

from run_hotpotqa_experiments import run_hotpotqa_experiment

results = run_hotpotqa_experiment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=1,  # Just 1 sample for debugging
    num_agents=3,
    model_identifier="Qwen/Qwen2.5-7B-Instruct",
    use_decomposer=True,
    use_rag=True,
    use_dynamic_credit=True,
    use_v3=True,
    use_iterative=True,
    output_dir=Path("results/debug_iter")
)
EOF

# Use conda to run in vlm-anchor environment
conda run -n vlm-anchor python /tmp/test_iter_debug.py 2>&1 | grep -A 20 "Using Iterative"
