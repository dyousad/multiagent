# RAG Components Installation Guide

## Required Dependencies

The new RAG components require the following Python packages:

```bash
pip install sentence-transformers faiss-cpu numpy
```

For GPU acceleration (optional):
```bash
pip install sentence-transformers faiss-gpu numpy
```

## Verification

After installation, run the test script:
```bash
python scripts/test_rag_components.py
```

All tests should pass with ✓ marks.

## Quick Start

1. Install dependencies:
   ```bash
   pip install sentence-transformers faiss-cpu numpy
   ```

2. Prepare a corpus (optional, or use HotpotQA context):
   ```python
   # The system can extract corpus from HotpotQA data automatically
   ```

3. Run a small experiment:
   ```bash
   python scripts/run_hotpotqa_experiments.py
   ```

## Component Overview

### New Files Created
- `src/retrieval_manager.py` - Manages document retrieval with FAISS
- `src/retriever_agent.py` - Agent for evidence retrieval
- `src/evidence_verifier_agent.py` - Agent for evidence verification

### Updated Files
- `src/controller.py` - Added `run_hotpotqa_pipeline()` method
- `scripts/run_hotpotqa_experiments.py` - Integrated RAG components

### Existing Files (Already Complete)
- `src/decomposer_agent.py` - Question decomposition
- `src/reward_manager.py` - Dynamic credit allocation

## Usage Example

```python
from retriever_agent import RetrieverAgent
from evidence_verifier_agent import EvidenceVerifierAgent
from controller import MultiAgentController

# Create RAG agents
retriever = RetrieverAgent(
    agent_id="retriever",
    retriever_config={"top_k": 10},
    top_k=5,
    rerank=True
)

verifier = EvidenceVerifierAgent(
    agent_id="verifier",
    model_identifier="deepseek-ai/DeepSeek-V3"
)

# Use in controller
controller = MultiAgentController(
    agents=[retriever, verifier, ...],
    environment=env,
    use_decomposer=True,
    decomposer_agent=decomposer
)

# Run RAG pipeline
task = {"question": "Your complex question here"}
result = await controller.run_hotpotqa_pipeline(task)
```

## Troubleshooting

### ImportError: No module named 'sentence_transformers'
```bash
pip install sentence-transformers
```

### ImportError: No module named 'faiss'
```bash
pip install faiss-cpu
```

### Memory Issues with Large Corpus
- Use a smaller sentence transformer model (e.g., "all-MiniLM-L6-v2")
- Process corpus in batches
- Use FAISS indexing for efficient search

## Next Steps

1. Install dependencies
2. Run test script to verify installation
3. Run small-scale HotpotQA experiment (10 samples)
4. Analyze results in `results/hotpotqa/`
5. Scale up to full dataset if needed
