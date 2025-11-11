"""Simple test script to verify the installation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from agent import Agent, Message
        print("✓ agent module imported")
    except Exception as e:
        print(f"✗ Failed to import agent: {e}")
        return False

    try:
        from environment import Environment, TaskResult
        print("✓ environment module imported")
    except Exception as e:
        print(f"✗ Failed to import environment: {e}")
        return False

    try:
        from controller import MultiAgentController
        print("✓ controller module imported")
    except Exception as e:
        print(f"✗ Failed to import controller: {e}")
        return False

    try:
        from shapley import compute_shapley
        print("✓ shapley module imported")
    except Exception as e:
        print(f"✗ Failed to import shapley: {e}")
        return False

    try:
        from reward_manager import RewardManager
        print("✓ reward_manager module imported")
    except Exception as e:
        print(f"✗ Failed to import reward_manager: {e}")
        return False

    return True


def test_shapley_computation():
    """Test Shapley value computation."""
    print("\nTesting Shapley computation...")

    from shapley import compute_shapley

    # Simple test case
    agents = ["agent_0", "agent_1", "agent_2"]
    contributions = {
        "agent_0": 0.5,
        "agent_1": 0.3,
        "agent_2": 0.2,
    }

    shapley_values = compute_shapley(
        contributions=contributions,
        agents=agents,
        use_monte_carlo=True,
        num_samples=100
    )

    print(f"Shapley values: {shapley_values}")

    # Check that values sum to 1
    total = sum(shapley_values.values())
    assert abs(total - 1.0) < 0.01, f"Shapley values should sum to 1, got {total}"
    print("✓ Shapley values sum to 1.0")

    return True


def test_environment():
    """Test environment creation."""
    print("\nTesting environment...")

    from environment import Environment

    env = Environment("Test task")
    obs = env.get_observation("agent_0")

    assert obs["task"] == "Test task"
    assert obs["agent_id"] == "agent_0"
    print("✓ Environment works correctly")

    return True


def test_agent():
    """Test agent creation."""
    print("\nTesting agent...")

    from agent import Agent, Message

    agent = Agent("test_agent", role="tester")
    msg = agent.communicate("Hello", receiver_id="agent_1")

    assert msg.sender_id == "test_agent"
    assert msg.receiver_id == "agent_1"
    assert msg.content == "Hello"
    print("✓ Agent works correctly")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Multi-Agent System Test Suite")
    print("="*60 + "\n")

    tests = [
        test_imports,
        test_shapley_computation,
        test_environment,
        test_agent,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("="*60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
