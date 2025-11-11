"""Visualization script for experiment results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(summary_file: Path) -> List[Dict]:
    """Load experiment summary.

    Parameters
    ----------
    summary_file : Path
        Path to summary.json file.

    Returns
    -------
    List[Dict]
        Experiment results.
    """
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_reward_comparison(all_results: List[Dict], output_dir: Path):
    """Plot comparison of reward methods across tasks.

    Parameters
    ----------
    all_results : List[Dict]
        Results from all experiments.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["shapley", "uniform", "proportional"]
    metrics = ["fairness_index", "variance", "mean_reward"]

    num_tasks = len(all_results)
    task_indices = np.arange(num_tasks)

    # Plot each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        width = 0.25
        x_positions = task_indices

        for i, method in enumerate(methods):
            values = []
            for task_result in all_results:
                if method in task_result['results'] and 'statistics' in task_result['results'][method]:
                    value = task_result['results'][method]['statistics'].get(metric, 0.0)
                    values.append(value)
                else:
                    values.append(0.0)

            ax.bar(x_positions + i * width, values, width, label=method.capitalize())

        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14)
        ax.set_xticks(x_positions + width)
        ax.set_xticklabels([f'Task {i+1}' for i in range(num_tasks)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_comparison.png', dpi=300)
        print(f"Saved plot: {output_dir / f'{metric}_comparison.png'}")
        plt.close()


def plot_reward_distribution(all_results: List[Dict], output_dir: Path):
    """Plot reward distribution for each method.

    Parameters
    ----------
    all_results : List[Dict]
        Results from all experiments.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["shapley", "uniform", "proportional"]

    for task_idx, task_result in enumerate(all_results):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for method_idx, method in enumerate(methods):
            ax = axes[method_idx]

            if method in task_result['results'] and 'rewards' in task_result['results'][method]:
                rewards = task_result['results'][method]['rewards']
                agent_ids = list(rewards.keys())
                reward_values = list(rewards.values())

                ax.bar(agent_ids, reward_values, color=f'C{method_idx}')
                ax.set_xlabel('Agent', fontsize=10)
                ax.set_ylabel('Reward', fontsize=10)
                ax.set_title(f'{method.capitalize()}', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(axis='y', alpha=0.3)

        task_name = task_result['task'][:40] + "..." if len(task_result['task']) > 40 else task_result['task']
        fig.suptitle(f'Task {task_idx+1}: {task_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'task_{task_idx+1}_rewards.png', dpi=300)
        print(f"Saved plot: {output_dir / f'task_{task_idx+1}_rewards.png'}")
        plt.close()


def plot_aggregate_metrics(all_results: List[Dict], output_dir: Path):
    """Plot aggregate metrics across all tasks.

    Parameters
    ----------
    all_results : List[Dict]
        Results from all experiments.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["shapley", "uniform", "proportional"]
    metrics = ["fairness_index", "variance", "std_deviation"]

    # Aggregate data
    aggregated = {method: {metric: [] for metric in metrics} for method in methods}

    for task_result in all_results:
        for method in methods:
            if method in task_result['results'] and 'statistics' in task_result['results'][method]:
                stats = task_result['results'][method]['statistics']
                for metric in metrics:
                    if metric in stats:
                        aggregated[method][metric].append(stats[metric])

    # Calculate means
    mean_values = {
        method: {
            metric: np.mean(values) if values else 0.0
            for metric, values in method_data.items()
        }
        for method, method_data in aggregated.items()
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        values = [mean_values[method][metric] for method in methods]
        colors = ['C0', 'C1', 'C2']

        ax.bar(methods, values, color=colors)
        ax.set_xlabel('Method', fontsize=10)
        ax.set_ylabel(f'Mean {metric.replace("_", " ").title()}', fontsize=10)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Aggregate Metrics Across All Tasks', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_metrics.png', dpi=300)
    print(f"Saved plot: {output_dir / 'aggregate_metrics.png'}")
    plt.close()


def main():
    """Main function for plotting results."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--summary", type=str, default="results/experiments/summary.json",
                        help="Path to summary.json file")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                        help="Output directory for plots")

    args = parser.parse_args()

    summary_file = Path(args.summary)
    output_dir = Path(args.output_dir)

    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)

    print(f"Loading results from {summary_file}...")
    all_results = load_summary(summary_file)

    print(f"Creating plots in {output_dir}...")
    plot_reward_comparison(all_results, output_dir)
    plot_reward_distribution(all_results, output_dir)
    plot_aggregate_metrics(all_results, output_dir)

    print("\nAll plots created successfully!")


if __name__ == "__main__":
    main()
