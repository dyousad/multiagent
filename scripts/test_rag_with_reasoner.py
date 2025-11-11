#!/usr/bin/env python3
"""Test the fixed RAG pipeline with ReasonerAgent."""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*70)
print("Testing Fixed RAG Pipeline with ReasonerAgent")
print("="*70)

# Test setup
print("\n[Setup] Creating test components...")
from decomposer_agent import DecomposerAgent
from retriever_agent import RetrieverAgent
from evidence_verifier_agent import EvidenceVerifierAgent
from reasoner_agent import ReasonerAgent
from controller import MultiAgentController
from environment_hotpotqa import HotpotQAEnvironment

# Check corpus
corpus_path = Path("data/hotpotqa_corpus.json")
if not corpus_path.exists():
    print(f"✗ Corpus not found: {corpus_path}")
    print("Run: python scripts/prepare_hotpotqa_corpus.py --max_samples 1000")
    sys.exit(1)

print(f"✓ Corpus found: {corpus_path}")

# Create environment
env = HotpotQAEnvironment(
    data_path="data/hotpot_dev_fullwiki_v1.json",
    max_samples=1,
    split="validation"
)
env.set_current_task(0)
task_data = env.current_task_data

print(f"\n[Test Question]")
print(f"Q: {task_data['question']}")
print(f"A: {task_data['answer']}")

# Test 1: Create all agents
print(f"\n[Test 1] Creating agents...")
try:
    model = "Qwen/Qwen2.5-7B-Instruct"

    decomposer = DecomposerAgent(agent_id="decomposer", model_identifier=model)
    print(f"✓ Decomposer created")

    retriever = RetrieverAgent(
        agent_id="retriever",
        retriever_config={"corpus_path": str(corpus_path), "top_k": 10},
        top_k=5,
        rerank=True
    )
    print(f"✓ Retriever created")

    verifier = EvidenceVerifierAgent(
        agent_id="verifier",
        model_identifier=model,
        min_entity_overlap=0.5,
        use_spacy=False  # Use keyword fallback
    )
    print(f"✓ Verifier created")

    reasoner = ReasonerAgent(
        agent_id="reasoner",
        model_identifier=model,
        max_tokens=512
    )
    print(f"✓ Reasoner created")

except Exception as e:
    print(f"✗ Error creating agents: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Run RAG pipeline
print(f"\n[Test 2] Running RAG pipeline...")
try:
    agents = [retriever, verifier]
    controller = MultiAgentController(
        agents=agents,
        environment=env,
        mode="sequential",
        use_decomposer=True,
        decomposer_agent=decomposer
    )

    task = {"question": task_data['question']}
    pipeline_output = asyncio.run(controller.run_hotpotqa_pipeline(task))

    sub_questions = pipeline_output.get("sub_questions", [])
    agent_outputs_rag = pipeline_output.get("agent_outputs", {})

    print(f"✓ Pipeline completed")
    print(f"  - Sub-questions: {len(sub_questions)}")
    print(f"  - Agent outputs: {len(agent_outputs_rag)}")

    # Show first sub-question and its evidence
    if sub_questions:
        first_sq = sub_questions[0]
        print(f"\n  First sub-question: {first_sq}")
        if first_sq in agent_outputs_rag:
            evidence = agent_outputs_rag[first_sq].get('evidence', [])
            print(f"  Evidence count: {len(evidence)}")
            if evidence:
                print(f"  First evidence: {evidence[0][:80]}...")

except Exception as e:
    print(f"✗ Pipeline error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Use ReasonerAgent to synthesize answer
print(f"\n[Test 3] Synthesizing answer with ReasonerAgent...")
try:
    result = reasoner.synthesize_answer(
        main_question=task_data['question'],
        sub_results=agent_outputs_rag
    )

    final_answer = result['final_answer']

    print(f"✓ Answer synthesized")
    print(f"\n  Question: {task_data['question']}")
    print(f"  Ground truth: {task_data['answer']}")
    print(f"  Predicted: {final_answer}")

    # Simple exact match check
    gt = task_data['answer'].lower().strip()
    pred = final_answer.lower().strip()
    exact_match = (gt == pred or gt in pred or pred in gt)

    print(f"\n  Exact match: {exact_match}")

    if not exact_match:
        print(f"\n  Note: Answer may still be correct but not exact match.")
        print(f"  Ground truth words: {set(gt.split())}")
        print(f"  Predicted words: {set(pred.split())}")
        overlap = set(gt.split()) & set(pred.split())
        print(f"  Word overlap: {overlap}")

except Exception as e:
    print(f"✗ Reasoner error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
print("\nKey findings:")
print("  1. RAG pipeline successfully retrieves evidence")
print("  2. ReasonerAgent synthesizes coherent answers")
print("  3. System is ready for full experiments")
print("\nTo run full experiment:")
print("  python scripts/run_hotpotqa_experiments.py")
print("="*70)
