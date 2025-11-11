#!/usr/bin/env python3
"""Quick test for RAG integration with HotpotQA."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*70)
print("Testing RAG Integration with HotpotQA")
print("="*70)

# Step 1: Check if corpus exists
print("\n[Step 1] Checking corpus file...")
corpus_path = Path("data/hotpotqa_corpus.json")
if corpus_path.exists():
    print(f"✓ Corpus found: {corpus_path}")
    import json
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"  - Documents in corpus: {len(corpus)}")
else:
    print(f"✗ Corpus not found: {corpus_path}")
    print("  Please run: python scripts/prepare_hotpotqa_corpus.py --max_samples 100")
    sys.exit(1)

# Step 2: Test RAG components with corpus
print("\n[Step 2] Testing RAG components with corpus...")
try:
    from retrieval_manager import RetrievalManager
    from retriever_agent import RetrieverAgent
    from evidence_verifier_agent import EvidenceVerifierAgent

    print("  Loading retrieval manager (this may take a moment)...")
    rm = RetrievalManager(
        corpus_path=str(corpus_path),
        model_name="all-MiniLM-L6-v2",
        top_k=5
    )
    print(f"✓ RetrievalManager loaded with {len(rm.corpus)} documents")

    # Test retrieval
    test_query = "Who directed the movie?"
    results = rm.retrieve(test_query, top_k=3)
    print(f"\n  Test query: '{test_query}'")
    print(f"  Retrieved {len(results)} results:")
    for i, result in enumerate(results[:2], 1):
        print(f"    {i}. {result[:80]}...")

except ImportError as e:
    print(f"⚠ Skipping (missing dependencies): {e}")
    print("  Install: pip install sentence-transformers faiss-cpu")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test RetrieverAgent
print("\n[Step 3] Testing RetrieverAgent...")
try:
    retriever = RetrieverAgent(
        agent_id="test_retriever",
        retriever_config={"corpus_path": str(corpus_path), "top_k": 10},
        top_k=5,
        rerank=True
    )
    print(f"✓ RetrieverAgent created")

    # Test retrieval
    result = retriever.retrieve_evidence("What is the capital of France?")
    print(f"  Retrieved {result['count']} evidence passages")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test EvidenceVerifierAgent
print("\n[Step 4] Testing EvidenceVerifierAgent...")
try:
    verifier = EvidenceVerifierAgent(
        agent_id="test_verifier",
        model_identifier="deepseek-ai/DeepSeek-V3",
        max_tokens=256
    )
    print(f"✓ EvidenceVerifierAgent created")

    # Test verification
    test_evidence = ["Paris is the capital of France.", "France is in Europe."]
    result = verifier.verify_evidence(
        "What is the capital of France?",
        test_evidence
    )
    print(f"  Verification: {result['verified']}")
    print(f"  Keywords matched: {result['matched_keywords']}/{result['total_keywords']}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test async pipeline
print("\n[Step 5] Testing async pipeline...")
try:
    import asyncio
    from controller import MultiAgentController
    from environment_hotpotqa import HotpotQAEnvironment
    from decomposer_agent import DecomposerAgent

    # Create minimal environment
    env = HotpotQAEnvironment(
        data_path="data/hotpot_dev_fullwiki_v1.json",
        max_samples=1,
        split="validation"
    )

    # Create decomposer
    decomposer = DecomposerAgent(
        agent_id="test_decomposer",
        model_identifier="deepseek-ai/DeepSeek-V3"
    )

    # Create controller with RAG agents
    controller = MultiAgentController(
        agents=[retriever, verifier],
        environment=env,
        mode="sequential",
        use_decomposer=True,
        decomposer_agent=decomposer
    )

    print("  Running async pipeline...")
    task = {"question": "Were Scott Derrickson and Ed Wood of the same nationality?"}
    result = asyncio.run(controller.run_hotpotqa_pipeline(task))

    print(f"✓ Pipeline completed")
    print(f"  Sub-questions: {len(result['sub_questions'])}")
    print(f"  Agent outputs: {len(result['agent_outputs'])}")

    if result['sub_questions']:
        print(f"\n  Example sub-question: {result['sub_questions'][0]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All RAG integration tests passed!")
print("="*70)
print("\nYou can now run:")
print("  python scripts/run_hotpotqa_experiments.py")
print("="*70)
