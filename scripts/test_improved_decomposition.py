"""Test script for improved question decomposition and reasoning."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decomposer_agent import DecomposerAgent
from reasoner_agent import ReasonerAgent


def test_decomposer_improvements():
    """Test improved decomposer that avoids pronouns."""
    print("="*70)
    print("Testing Improved Decomposer Agent")
    print("="*70)

    decomposer = DecomposerAgent(
        agent_id="decomposer_test",
        model_identifier="deepseek-ai/DeepSeek-V3"
    )

    # Test cases that previously failed
    test_questions = [
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "The director of the romantic comedy Big Stone Gap is based in what New York city?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {question}")
        print(f"{'='*70}")

        result = decomposer.decompose_question(question)

        print("\nSub-questions:")
        for j, sq in enumerate(result["sub_questions"], 1):
            print(f"{j}. {sq}")
            # Check for pronouns
            pronouns = ["this", "that", "it", "he", "she", "they", "his", "her", "their"]
            found_pronouns = [p for p in pronouns if p in sq.lower().split()]
            if found_pronouns:
                print(f"   ⚠️  WARNING: Found pronouns: {found_pronouns}")
            else:
                print(f"   ✓ No pronouns detected")


def test_reasoner_improvements():
    """Test improved reasoner that outputs concise answers."""
    print("\n\n" + "="*70)
    print("Testing Improved Reasoner Agent")
    print("="*70)

    reasoner = ReasonerAgent(
        agent_id="reasoner_test",
        model_identifier="Qwen/Qwen2.5-7B-Instruct"
    )

    # Test yes/no question detection
    yes_no_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "Is this a test?",
    ]

    print("\nYes/No Question Detection:")
    for q in yes_no_questions:
        is_yn = reasoner._is_yes_no_question(q)
        print(f"  {q}")
        print(f"  → Detected as yes/no: {is_yn}")

    # Test answer extraction
    print("\n" + "="*70)
    print("Testing Answer Extraction")
    print("="*70)

    test_cases = [
        {
            "response": "No. The evidence does not support Scott Derrickson and Ed Wood being of the same nationality.",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "expected": "no"
        },
        {
            "response": "Based on the evidence, she held the position of Chief of Protocol.",
            "question": "What government position did she hold?",
            "expected": "Chief of Protocol"
        },
        {
            "response": "The series in question is Animorphs, a science fantasy series.",
            "question": "What series is it?",
            "expected": "Animorphs"
        },
        {
            "response": "yes",
            "question": "Is this correct?",
            "expected": "yes"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Question: {test['question']}")
        print(f"  Raw response: {test['response']}")
        extracted = reasoner._extract_answer(test['response'], test['question'])
        print(f"  Extracted: '{extracted}'")
        print(f"  Expected: '{test['expected']}'")
        match = extracted.lower().strip() == test['expected'].lower().strip()
        print(f"  Status: {'✓ PASS' if match else '✗ FAIL'}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("IMPROVED QUESTION DECOMPOSITION & REASONING TEST SUITE")
    print("="*70 + "\n")

    try:
        test_decomposer_improvements()
    except Exception as e:
        print(f"\n✗ Decomposer test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_reasoner_improvements()
    except Exception as e:
        print(f"\n✗ Reasoner test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
