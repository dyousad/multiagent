#!/usr/bin/env python
"""Test script to verify HotpotQA integration is working correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")

    try:
        from environment_hotpotqa import HotpotQAEnvironment
        print("✓ HotpotQAEnvironment imported successfully")
    except Exception as e:
        print(f"✗ Failed to import HotpotQAEnvironment: {e}")
        return False

    try:
        from decomposer_agent import DecomposerAgent
        print("✓ DecomposerAgent imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DecomposerAgent: {e}")
        return False

    try:
        from evaluation import (
            multi_hop_accuracy,
            token_f1_score,
            fairness_index,
            credit_entropy
        )
        print("✓ Evaluation metrics imported successfully")
    except Exception as e:
        print(f"✗ Failed to import evaluation metrics: {e}")
        return False

    try:
        from reward_manager import RewardManager
        print("✓ RewardManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import RewardManager: {e}")
        return False

    return True


def test_hotpotqa_environment():
    """Test HotpotQA environment basic functionality."""
    print("\nTesting HotpotQA environment...")

    from environment_hotpotqa import HotpotQAEnvironment

    # Check if data file exists
    data_path = Path("data/hotpot_dev_fullwiki_v1.json")
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        print("  Please download HotpotQA data to the data/ directory")
        return False

    try:
        # Create environment
        env = HotpotQAEnvironment(
            data_path=str(data_path),
            max_samples=5,
            split="validation"
        )
        print(f"✓ Loaded {env.get_num_samples()} samples")

        # Test getting a task
        task = env.get_task(0)
        print(f"✓ Retrieved task: {task['question'][:50]}...")

        # Test evaluation
        env.set_current_task(0)
        test_responses = {"agent_0": "test answer"}
        result = env.evaluate(test_responses)
        print(f"✓ Evaluation completed: EM={result.metadata['exact_match']}, F1={result.metadata['f1_score']:.3f}")

        return True
    except Exception as e:
        print(f"✗ HotpotQA environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decomposer_agent():
    """Test DecomposerAgent basic functionality."""
    print("\nTesting DecomposerAgent...")

    from decomposer_agent import DecomposerAgent

    try:
        # Create decomposer (this will fail if API not configured, but import should work)
        decomposer = DecomposerAgent(
            agent_id="test_decomposer",
            model_identifier="deepseek-ai/DeepSeek-V3"
        )
        print("✓ DecomposerAgent created successfully")

        # Test parsing
        sample_text = """1. What is the capital of France?
2. What is the population of that city?
3. Is it larger than London?"""

        sub_questions = decomposer.parse_sub_questions(sample_text)
        print(f"✓ Parsed {len(sub_questions)} sub-questions")
        for i, sq in enumerate(sub_questions):
            print(f"  {i+1}. {sq}")

        return True
    except Exception as e:
        print(f"✗ DecomposerAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_credit_allocation():
    """Test dynamic credit allocation functionality."""
    print("\nTesting dynamic credit allocation...")

    from reward_manager import RewardManager

    try:
        reward_manager = RewardManager(base_reward=1.0)

        # Test dynamic credit update
        agent_outputs = {
            "agent_0": "The answer is Paris",
            "agent_1": "I agree, Paris is correct",
            "agent_2": "Yes, Paris"
        }
        final_answer = "Paris"
        ground_truth = "Paris"

        credits = reward_manager.update_credits_dynamic(
            agent_outputs=agent_outputs,
            final_answer=final_answer,
            ground_truth=ground_truth
        )
        print(f"✓ Dynamic credits computed: {credits}")

        # Test entropy calculation
        entropy = reward_manager.get_credit_entropy(credits)
        print(f"✓ Credit entropy calculated: {entropy:.3f}")

        return True
    except Exception as e:
        print(f"✗ Dynamic credit allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")

    from evaluation import (
        multi_hop_accuracy,
        token_f1_score,
        fairness_index,
        jain_fairness_index,
        credit_entropy
    )

    try:
        # Test multi-hop accuracy
        predictions = ["Paris", "London", "Berlin"]
        gold_answers = ["Paris", "london", "Berlin"]  # Test case-insensitive
        accuracy = multi_hop_accuracy(predictions, gold_answers)
        print(f"✓ Multi-hop accuracy: {accuracy:.3f}")

        # Test F1 score
        pred = "The capital of France is Paris"
        truth = "Paris is the capital of France"
        f1 = token_f1_score(pred, truth)
        print(f"✓ Token F1 score: {f1:.3f}")

        # Test fairness indices
        credits = [0.3, 0.4, 0.3]
        gini = fairness_index(credits)
        jain = jain_fairness_index(credits)
        print(f"✓ Gini fairness: {gini:.3f}, Jain fairness: {jain:.3f}")

        # Test entropy
        credit_dict = {"agent_0": 0.3, "agent_1": 0.4, "agent_2": 0.3}
        ent = credit_entropy(credit_dict)
        print(f"✓ Credit entropy: {ent:.3f}")

        return True
    except Exception as e:
        print(f"✗ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("HotpotQA Integration Test Suite")
    print("="*70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("HotpotQA Environment", test_hotpotqa_environment()))
    results.append(("DecomposerAgent", test_decomposer_agent()))
    results.append(("Dynamic Credit Allocation", test_dynamic_credit_allocation()))
    results.append(("Evaluation Metrics", test_evaluation_metrics()))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! HotpotQA integration is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
