"""Experiment runner for comparing reward allocation methods."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import run_experiment


def run_all_experiments(
    tasks: list[str],
    num_agents: int = 3,
    model_identifier: str = "deepseek-ai/DeepSeek-V3",
    output_dir: Path = Path("results/experiments")
):
    """Run experiments comparing all reward methods.

    Parameters
    ----------
    tasks : list[str]
        List of tasks to evaluate.
    num_agents : int
        Number of agents.
    model_identifier : str
        Model identifier for LLM API.
    output_dir : Path
        Output directory for results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_methods = ["shapley", "uniform", "proportional"]
    all_results = []

    for i, task in enumerate(tasks):
        print(f"\n{'#'*70}")
        print(f"# Task {i+1}/{len(tasks)}: {task[:50]}...")
        print(f"{'#'*70}\n")

        task_results = {}

        for method in reward_methods:
            try:
                result = run_experiment(
                    task_description=task,
                    num_agents=num_agents,
                    model_identifier=model_identifier,
                    reward_method=method,
                    mode="sequential",
                    output_dir=output_dir / f"task_{i}_{method}"
                )
                task_results[method] = result
            except Exception as e:
                print(f"Error in {method} experiment: {e}")
                task_results[method] = {"error": str(e)}

        all_results.append({
            "task": task,
            "results": task_results
        })

    # Save aggregated results
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*70}\n")

    # Print comparison
    print_comparison(all_results)


def print_comparison(all_results: list[dict]):
    """Print comparison of reward methods.

    Parameters
    ----------
    all_results : list[dict]
        Results from all experiments.
    """
    print("\n" + "="*70)
    print("COMPARISON OF REWARD METHODS")
    print("="*70)

    for i, task_result in enumerate(all_results):
        print(f"\nTask {i+1}: {task_result['task'][:50]}...")
        print("-" * 70)

        methods = ["shapley", "uniform", "proportional"]
        metrics = ["fairness_index", "variance", "mean_reward"]

        # Print header
        print(f"{'Metric':<20} " + " ".join(f"{m:<15}" for m in methods))
        print("-" * 70)

        # Print metrics
        for metric in metrics:
            row = f"{metric:<20} "
            for method in methods:
                if method in task_result['results'] and 'statistics' in task_result['results'][method]:
                    value = task_result['results'][method]['statistics'].get(metric, 0.0)
                    row += f"{value:<15.4f}"
                else:
                    row += f"{'N/A':<15}"
            print(row)

    print("="*70 + "\n")


def main():
    """Main function for experiment runner."""
    # Define evaluation tasks
    tasks = [
        "Write a Python function to calculate the factorial of a number recursively.",
        "Implement a binary search algorithm in Python.",
        "Design a simple REST API for a todo list application using Flask.",
        "Explain the concept of gradient descent in machine learning.",
        "Write a function to find the longest common subsequence of two strings."
    ]

    run_all_experiments(
        tasks=tasks,
        num_agents=3,
        model_identifier="Qwen/Qwen2.5-7B-Instruct",
        output_dir=Path("results/experiments")
    )


if __name__ == "__main__":
    main()
