#!/usr/bin/env python3
"""Debug script to analyze HotpotQA context parsing issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment_hotpotqa import HotpotQAEnvironment

def debug_context_parsing():
    """Debug HotpotQA context parsing to understand unknown answers."""
    print("🔍 Debugging HotpotQA Context Parsing")
    print("=" * 60)

    # Load environment
    env = HotpotQAEnvironment(
        data_path="data/hotpot_dev_fullwiki_v1.json",
        max_samples=5
    )

    for idx in range(min(3, env.get_num_samples())):
        print(f"\n{'='*60}")
        print(f"Sample {idx}")
        print(f"{'='*60}")

        env.set_current_task(idx)
        task_data = env.current_task_data

        print(f"Question: {task_data['question']}")
        print(f"Answer: {task_data['answer']}")

        # Debug context parsing
        if 'context' in task_data:
            context = task_data['context']
            print(f"\nOriginal context type: {type(context)}")
            print(f"Context length: {len(context)}")

            # Test our parsing logic
            context_evidence = []
            for i, ctx_item in enumerate(context[:3]):
                print(f"\nContext item {i}:")
                print(f"  Type: {type(ctx_item)}")
                print(f"  Length: {len(ctx_item) if hasattr(ctx_item, '__len__') else 'N/A'}")
                print(f"  Content preview: {str(ctx_item)[:200]}...")

                # Apply our parsing logic
                if isinstance(ctx_item, (list, tuple)) and len(ctx_item) >= 2:
                    title, sents = ctx_item[0], ctx_item[1:]
                    print(f"  Title: {title}")
                    print(f"  Sentences type: {type(sents)}")
                    print(f"  Sentences length: {len(sents) if hasattr(sents, '__len__') else 'N/A'}")

                    if isinstance(sents, list):
                        print(f"  First sentence: {sents[0][:100] if sents else 'Empty'}...")
                        evidence_item = f"{title}: {' '.join(sents[:3])}"
                        context_evidence.append(evidence_item)
                        print(f"  Generated evidence: {evidence_item[:150]}...")
                    else:
                        evidence_item = f"{title}: {str(sents)[:300]}"
                        context_evidence.append(evidence_item)
                        print(f"  Generated evidence (non-list): {evidence_item[:150]}...")
                else:
                    evidence_item = str(ctx_item)[:300]
                    context_evidence.append(evidence_item)
                    print(f"  Generated evidence (fallback): {evidence_item[:150]}...")

            print(f"\n📊 Final context_evidence:")
            for i, ev in enumerate(context_evidence):
                print(f"  [{i}]: {ev[:100]}...")

            print(f"\n🔍 Evidence paths (first 100 chars):")
            evidence_paths = [ev[:100] for ev in context_evidence[:4]]
            for i, ep in enumerate(evidence_paths):
                print(f"  [{i}]: '{ep}'")

if __name__ == "__main__":
    debug_context_parsing()