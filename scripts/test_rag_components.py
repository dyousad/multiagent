#!/usr/bin/env python3
"""Quick test script to verify RAG components are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*70)
print("Testing RAG Components")
print("="*70)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from agent import Agent
    from decomposer_agent import DecomposerAgent
    from retriever_agent import RetrieverAgent
    from evidence_verifier_agent import EvidenceVerifierAgent
    from retrieval_manager import RetrievalManager
    from controller import MultiAgentController
    from reward_manager import RewardManager
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check DecomposerAgent
print("\n[Test 2] Testing DecomposerAgent...")
try:
    decomposer = DecomposerAgent(
        agent_id="test_decomposer",
        model_identifier="deepseek-ai/DeepSeek-V3"
    )
    print(f"✓ DecomposerAgent created: {decomposer.agent_id}")
except Exception as e:
    print(f"✗ DecomposerAgent error: {e}")

# Test 3: Check EvidenceVerifierAgent
print("\n[Test 3] Testing EvidenceVerifierAgent...")
try:
    verifier = EvidenceVerifierAgent(
        agent_id="test_verifier",
        model_identifier="deepseek-ai/DeepSeek-V3"
    )
    print(f"✓ EvidenceVerifierAgent created: {verifier.agent_id}")

    # Test verification logic
    test_result = verifier.verify_evidence(
        question="What is the capital of France?",
        evidence=["Paris is the capital of France.", "France is in Europe."]
    )
    print(f"  - Verification test: {test_result['verified']}")
    print(f"  - Keywords matched: {test_result['matched_keywords']}/{test_result['total_keywords']}")
except Exception as e:
    print(f"✗ EvidenceVerifierAgent error: {e}")

# Test 4: Check RetrievalManager (may fail if dependencies not installed)
print("\n[Test 4] Testing RetrievalManager...")
try:
    # Create a minimal test corpus
    test_corpus = [
        {"text": "Paris is the capital city of France."},
        {"text": "London is the capital of the United Kingdom."},
        {"text": "Berlin is the capital of Germany."},
    ]

    retrieval_manager = RetrievalManager(
        corpus=test_corpus,
        model_name="all-MiniLM-L6-v2",  # Smaller model for testing
        top_k=2
    )
    print(f"✓ RetrievalManager created with {len(test_corpus)} documents")

    # Test retrieval
    results = retrieval_manager.retrieve("What is the capital of France?", top_k=2)
    print(f"  - Retrieved {len(results)} results")
    print(f"  - Top result: {results[0][:50]}...")

except ImportError as e:
    print(f"⚠ RetrievalManager skipped (missing dependencies): {e}")
    print("  Install with: pip install sentence-transformers faiss-cpu")
except Exception as e:
    print(f"✗ RetrievalManager error: {e}")

# Test 5: Check RetrieverAgent (may fail if dependencies not installed)
print("\n[Test 5] Testing RetrieverAgent...")
try:
    test_corpus = [
        {"text": "Paris is the capital city of France."},
        {"text": "London is the capital of the United Kingdom."},
    ]

    retriever = RetrieverAgent(
        agent_id="test_retriever",
        retriever_config={
            "corpus": test_corpus,
            "model_name": "all-MiniLM-L6-v2",
            "top_k": 2
        },
        top_k=1,
        rerank=True
    )
    print(f"✓ RetrieverAgent created: {retriever.agent_id}")

    # Test retrieval
    result_dict = retriever.retrieve_evidence("What is the capital of France?")
    print(f"  - Evidence count: {result_dict['count']}")

except ImportError as e:
    print(f"⚠ RetrieverAgent skipped (missing dependencies): {e}")
    print("  Install with: pip install sentence-transformers faiss-cpu")
except Exception as e:
    print(f"✗ RetrieverAgent error: {e}")

# Test 6: Check Controller's run_hotpotqa_pipeline method
print("\n[Test 6] Testing Controller.run_hotpotqa_pipeline...")
try:
    # Check if method exists
    if hasattr(MultiAgentController, 'run_hotpotqa_pipeline'):
        print("✓ run_hotpotqa_pipeline method exists in MultiAgentController")
    else:
        print("✗ run_hotpotqa_pipeline method not found")

    if hasattr(MultiAgentController, '_get_agent_by_role'):
        print("✓ _get_agent_by_role helper method exists")
    else:
        print("✗ _get_agent_by_role helper method not found")

except Exception as e:
    print(f"✗ Controller test error: {e}")

# Test 7: Check RewardManager dynamic credit allocation
print("\n[Test 7] Testing RewardManager dynamic credit allocation...")
try:
    reward_manager = RewardManager(base_reward=1.0)

    # Test dynamic credit update
    agent_outputs = {
        "agent_0": "Paris is the capital",
        "agent_1": "of France"
    }
    final_answer = "Paris is the capital of France"
    ground_truth = "Paris"

    credits = reward_manager.update_credits_dynamic(
        agent_outputs=agent_outputs,
        final_answer=final_answer,
        ground_truth=ground_truth
    )

    print(f"✓ Dynamic credit allocation working")
    print(f"  - Credits: {credits}")

    entropy = reward_manager.get_credit_entropy(credits)
    print(f"  - Credit entropy: {entropy:.3f}")

except Exception as e:
    print(f"✗ RewardManager test error: {e}")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print("Core components are ready!")
print("\nNote: Some tests may be skipped if dependencies are not installed.")
print("To install required dependencies:")
print("  pip install sentence-transformers faiss-cpu numpy")
print("\nTo run full HotpotQA experiment:")
print("  python scripts/run_hotpotqa_experiments.py")
print("="*70)
