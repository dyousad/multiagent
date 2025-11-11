"""Main entrypoint for multi-agent experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from environment import Environment
from environment_hotpotqa import HotpotQAEnvironment
from llm_agent import LLMAgent
from controller import MultiAgentController
from reward_manager import RewardManager


def create_agents(
    num_agents: int,
    model_identifier: str,
    agent_roles: List[str] = None
) -> List[LLMAgent]:
    """Create a list of LLM agents.

    Parameters
    ----------
    num_agents : int
        Number of agents to create.
    model_identifier : str
        Model identifier for LLM API.
    agent_roles : List[str]
        List of roles for agents. If None, uses default roles.

    Returns
    -------
    List[LLMAgent]
        List of initialized agents.
    """
    if agent_roles is None:
        default_roles = ["planner", "coder", "reviewer", "tester", "documenter"]
        agent_roles = (default_roles * ((num_agents // len(default_roles)) + 1))[:num_agents]

    agents = []
    for i in range(num_agents):
        role = agent_roles[i] if i < len(agent_roles) else "assistant"
        system_prompt = f"You are a {role}. Your goal is to contribute to solving the task effectively."

        agent = LLMAgent(
            agent_id=f"agent_{i}",
            model_identifier=model_identifier,
            role=role,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.7
        )
        agents.append(agent)

    return agents


def evaluate_contributions(agents: List[LLMAgent], task_result) -> Dict[str, float]:
    """Evaluate agent contributions based on response quality.

    This is a simple heuristic. In practice, you might use:
    - LLM-based evaluation
    - Human evaluation
    - Task-specific metrics

    Parameters
    ----------
    agents : List[LLMAgent]
        List of agents.
    task_result : TaskResult
        Result of the task execution.

    Returns
    -------
    Dict[str, float]
        Contribution score for each agent.
    """
    contributions = {}

    for agent in agents:
        # Simple heuristic: contribution = response length normalized
        response = task_result.agent_responses.get(agent.agent_id, "")
        # Score based on response length and content
        score = len(response.split()) / 100.0  # Normalize by ~100 words
        score = min(score, 1.0)  # Cap at 1.0
        contributions[agent.agent_id] = score

    return contributions


def run_experiment(
    task_description: str,
    num_agents: int,
    model_identifier: str,
    reward_method: str = "shapley",
    mode: str = "sequential",
    output_dir: Path = Path("results"),
    env_type: str = "default",
    env_config: Dict = None
) -> Dict:
    """Run a single experiment.

    Parameters
    ----------
    task_description : str
        The task to solve.
    num_agents : int
        Number of agents.
    model_identifier : str
        Model identifier for LLM API.
    reward_method : str
        Reward allocation method: "shapley", "uniform", or "proportional".
    mode : str
        Execution mode: "parallel" or "sequential".
    output_dir : Path
        Directory to save results.
    env_type : str
        Environment type: "default" or "hotpotqa".
    env_config : Dict
        Environment configuration parameters.

    Returns
    -------
    Dict
        Experiment results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment:")
    print(f"  Task: {task_description[:50]}...")
    print(f"  Agents: {num_agents}")
    print(f"  Model: {model_identifier}")
    print(f"  Reward: {reward_method}")
    print(f"  Mode: {mode}")
    print(f"  Environment: {env_type}")
    print(f"{'='*60}\n")

    # Create environment based on type
    if env_type == "hotpotqa":
        env_config = env_config or {}
        env = HotpotQAEnvironment(
            data_path=env_config.get("data_path", "data/hotpot_dev_fullwiki_v1.json"),
            max_samples=env_config.get("max_samples", 100),
            split=env_config.get("split", "validation")
        )
        # For HotpotQA, task_description can be an index or we use the first sample
        if task_description.isdigit():
            env.set_current_task(int(task_description))
        else:
            env.set_current_task(0)  # Default to first sample
    else:
        env = Environment(task_description)

    agents = create_agents(num_agents, model_identifier)

    # Create controller
    controller = MultiAgentController(agents, env, mode=mode)

    # Run the task
    print("Running multi-agent system...")
    task_result = controller.run_sync()

    print(f"\nTask completed: {task_result.success}")
    print(f"Rounds: {task_result.metadata['rounds']}")

    # Evaluate contributions
    contributions = evaluate_contributions(agents, task_result)
    print(f"\nContributions: {contributions}")

    # Allocate rewards
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"rewards_{reward_method}.json"
    reward_manager = RewardManager(base_reward=1.0, log_file=log_file)

    if reward_method == "shapley":
        rewards = reward_manager.allocate_shapley_rewards(
            agents=agents,
            contributions=contributions,
            use_monte_carlo=True,
            num_samples=1000
        )
    elif reward_method == "uniform":
        rewards = reward_manager.allocate_uniform_rewards(agents)
    elif reward_method == "proportional":
        rewards = reward_manager.allocate_proportional_rewards(agents, contributions)
    else:
        raise ValueError(f"Unknown reward method: {reward_method}")

    print(f"\nRewards: {rewards}")

    # Get statistics
    stats = reward_manager.get_reward_statistics()
    print(f"\nReward statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results = {
        "task": task_description,
        "num_agents": num_agents,
        "model": model_identifier,
        "reward_method": reward_method,
        "mode": mode,
        "success": task_result.success,
        "rounds": task_result.metadata["rounds"],
        "agent_responses": task_result.agent_responses,
        "contributions": contributions,
        "rewards": rewards,
        "statistics": stats,
    }

    results_file = output_dir / f"results_{reward_method}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Multi-Agent LLM Collaboration with Shapley Rewards")
    parser.add_argument("--task", type=str, default="Write a Python function to calculate Fibonacci numbers",
                        help="Task description (or sample index for HotpotQA)")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="Model identifier (e.g., deepseek-ai/DeepSeek-V3, Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--reward", type=str, default="shapley",
                        choices=["shapley", "uniform", "proportional"],
                        help="Reward allocation method")
    parser.add_argument("--mode", type=str, default="sequential",
                        choices=["parallel", "sequential", "message_passing"],
                        help="Execution mode")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--env", type=str, default="default",
                        choices=["default", "hotpotqa"],
                        help="Environment type")
    parser.add_argument("--data_path", type=str, default="data/hotpot_dev_fullwiki_v1.json",
                        help="Path to HotpotQA data file")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of HotpotQA samples to load")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["validation", "train", "test"],
                        help="HotpotQA data split")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Prepare environment config
    env_config = {
        "data_path": args.data_path,
        "max_samples": args.max_samples,
        "split": args.split
    }

    try:
        results = run_experiment(
            task_description=args.task,
            num_agents=args.num_agents,
            model_identifier=args.model,
            reward_method=args.reward,
            mode=args.mode,
            output_dir=output_dir,
            env_type=args.env,
            env_config=env_config
        )
        print("\n" + "="*60)
        print("Experiment completed successfully!")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\nError during experiment: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
