"""Visualization script for HotpotQA experiment results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_hotpotqa_results(results_file: Path) -> Dict:
    """Load HotpotQA experiment results.

    Parameters
    ----------
    results_file : Path
        Path to hotpotqa_results.json file.

    Returns
    -------
    Dict
        Experiment results.
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_credit_entropy_comparison(results: Dict, output_dir: Path):
    """Plot comparison of static vs dynamic credit entropy.

    Parameters
    ----------
    results : Dict
        HotpotQA experiment results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    static_entropy = results['metrics']['credit_entropy_static']
    dynamic_entropy = results['metrics'].get('credit_entropy_dynamic', [])

    if not dynamic_entropy:
        print("Warning: No dynamic entropy data available")
        return

    sample_indices = np.arange(len(static_entropy))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sample_indices, static_entropy, marker='o', label='Static (Shapley)',
            linewidth=2, markersize=6, color='C0')
    ax.plot(sample_indices, dynamic_entropy, marker='s', label='Dynamic (Counterfactual)',
            linewidth=2, markersize=6, color='C1')

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Credit Entropy', fontsize=12)
    ax.set_title('Credit Entropy Comparison: Static vs Dynamic', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'credit_entropy_comparison.png', dpi=300)
    print(f"Saved plot: {output_dir / 'credit_entropy_comparison.png'}")
    plt.close()


def plot_performance_metrics(results: Dict, output_dir: Path):
    """Plot performance metrics (EM and F1).

    Parameters
    ----------
    results : Dict
        HotpotQA experiment results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_match = results['metrics']['exact_match']
    f1_scores = results['metrics']['f1_scores']

    sample_indices = np.arange(len(exact_match))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Exact match plot
    em_values = [1.0 if em else 0.0 for em in exact_match]
    ax1.bar(sample_indices, em_values, color='C2', alpha=0.7)
    ax1.axhline(y=np.mean(em_values), color='r', linestyle='--',
                label=f'Mean: {np.mean(em_values):.3f}', linewidth=2)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Exact Match (0 or 1)', fontsize=12)
    ax1.set_title('Exact Match Accuracy per Sample', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # F1 score plot
    ax2.plot(sample_indices, f1_scores, marker='o', linewidth=2,
             markersize=6, color='C3', label='F1 Score')
    ax2.axhline(y=np.mean(f1_scores), color='r', linestyle='--',
                label=f'Mean: {np.mean(f1_scores):.3f}', linewidth=2)
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score per Sample', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300)
    print(f"Saved plot: {output_dir / 'performance_metrics.png'}")
    plt.close()


def plot_credit_distribution(results: Dict, output_dir: Path):
    """Plot credit distribution across agents for a few samples.

    Parameters
    ----------
    results : Dict
        HotpotQA experiment results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = results['samples']
    num_samples_to_plot = min(5, len(samples))

    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(12, 4 * num_samples_to_plot))

    if num_samples_to_plot == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples_to_plot):
        sample = samples[i]

        # Static rewards
        static_rewards = sample.get('static_rewards', {})
        agents_static = list(static_rewards.keys())
        values_static = list(static_rewards.values())

        # Dynamic credits
        dynamic_credits = sample.get('dynamic_credits', {})
        agents_dynamic = list(dynamic_credits.keys())
        values_dynamic = list(dynamic_credits.values())

        # Plot static
        axes[i, 0].bar(agents_static, values_static, color='C0', alpha=0.7)
        axes[i, 0].set_ylabel('Reward', fontsize=10)
        axes[i, 0].set_title(f'Sample {i+1}: Static (Shapley)', fontsize=11)
        axes[i, 0].tick_params(axis='x', rotation=45)
        axes[i, 0].grid(axis='y', alpha=0.3)

        # Plot dynamic
        if dynamic_credits:
            axes[i, 1].bar(agents_dynamic, values_dynamic, color='C1', alpha=0.7)
            axes[i, 1].set_ylabel('Credit', fontsize=10)
            axes[i, 1].set_title(f'Sample {i+1}: Dynamic (Counterfactual)', fontsize=11)
            axes[i, 1].tick_params(axis='x', rotation=45)
            axes[i, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'credit_distribution.png', dpi=300)
    print(f"Saved plot: {output_dir / 'credit_distribution.png'}")
    plt.close()


def plot_aggregate_summary(results: Dict, output_dir: Path):
    """Plot aggregate summary metrics.

    Parameters
    ----------
    results : Dict
        HotpotQA experiment results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = results.get('aggregate', {})

    # Prepare data
    metrics_data = {
        'Exact Match\nAccuracy': aggregate.get('exact_match_accuracy', 0.0),
        'Average\nF1 Score': aggregate.get('average_f1', 0.0),
        'Static\nEntropy': aggregate.get('average_static_entropy', 0.0),
    }

    if 'average_dynamic_entropy' in aggregate:
        metrics_data['Dynamic\nEntropy'] = aggregate.get('average_dynamic_entropy', 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics_data.keys())
    metric_values = list(metrics_data.values())
    colors = ['C2', 'C3', 'C0', 'C1'][:len(metric_names)]

    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Aggregate Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(metric_values) * 1.2])

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_summary.png', dpi=300)
    print(f"Saved plot: {output_dir / 'aggregate_summary.png'}")
    plt.close()


def plot_entropy_histogram(results: Dict, output_dir: Path):
    """Plot histogram of entropy distributions.

    Parameters
    ----------
    results : Dict
        HotpotQA experiment results.
    output_dir : Path
        Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    static_entropy = results['metrics']['credit_entropy_static']
    dynamic_entropy = results['metrics'].get('credit_entropy_dynamic', [])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot static histogram
    ax.hist(static_entropy, bins=15, alpha=0.6, label='Static (Shapley)',
            color='C0', edgecolor='black')

    # Plot dynamic histogram if available
    if dynamic_entropy:
        ax.hist(dynamic_entropy, bins=15, alpha=0.6, label='Dynamic (Counterfactual)',
                color='C1', edgecolor='black')

    ax.set_xlabel('Credit Entropy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Credit Entropy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_histogram.png', dpi=300)
    print(f"Saved plot: {output_dir / 'entropy_histogram.png'}")
    plt.close()


def main():
    """Main function for plotting HotpotQA results."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot HotpotQA experiment results")
    parser.add_argument("--results", type=str, default="results/hotpotqa/hotpotqa_results.json",
                        help="Path to hotpotqa_results.json file")
    parser.add_argument("--output_dir", type=str, default="results/hotpotqa/plots",
                        help="Output directory for plots")

    args = parser.parse_args()

    results_file = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    print(f"Loading results from {results_file}...")
    results = load_hotpotqa_results(results_file)

    print(f"Creating plots in {output_dir}...")
    plot_credit_entropy_comparison(results, output_dir)
    plot_performance_metrics(results, output_dir)
    plot_credit_distribution(results, output_dir)
    plot_aggregate_summary(results, output_dir)
    plot_entropy_histogram(results, output_dir)

    print("\nAll HotpotQA plots created successfully!")


if __name__ == "__main__":
    main()
