"""Evaluation metrics for multi-hop reasoning tasks."""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import f1_score as sklearn_f1


def multi_hop_accuracy(predictions: List[str], gold_answers: List[str]) -> float:
    """Calculate exact match accuracy for multi-hop QA.

    Parameters
    ----------
    predictions : List[str]
        List of predicted answers.
    gold_answers : List[str]
        List of ground truth answers.

    Returns
    -------
    float
        Accuracy score (0.0 to 1.0).
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have the same length")

    if len(predictions) == 0:
        return 0.0

    matches = sum(
        p.strip().lower() == g.strip().lower()
        for p, g in zip(predictions, gold_answers)
    )

    return matches / len(predictions)


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score between prediction and ground truth.

    Parameters
    ----------
    prediction : str
        Predicted answer.
    ground_truth : str
        Ground truth answer.

    Returns
    -------
    float
        F1 score (0.0 to 1.0).
    """
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common_tokens = pred_tokens & truth_tokens

    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def average_f1_score(predictions: List[str], gold_answers: List[str]) -> float:
    """Calculate average token-level F1 score across multiple predictions.

    Parameters
    ----------
    predictions : List[str]
        List of predicted answers.
    gold_answers : List[str]
        List of ground truth answers.

    Returns
    -------
    float
        Average F1 score (0.0 to 1.0).
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("Predictions and gold answers must have the same length")

    if len(predictions) == 0:
        return 0.0

    f1_scores = [
        token_f1_score(p, g)
        for p, g in zip(predictions, gold_answers)
    ]

    return sum(f1_scores) / len(f1_scores)


def fairness_index(credits: List[float]) -> float:
    """Calculate Gini fairness index for credit distribution.

    The Gini coefficient measures inequality in a distribution.
    Lower values indicate more equal distribution (more fair).

    Parameters
    ----------
    credits : List[float]
        List of credit values for agents.

    Returns
    -------
    float
        Fairness index (0.0 to 1.0). Lower is more fair.
    """
    if len(credits) == 0:
        return 0.0

    credits_array = np.array(credits)
    mean = np.mean(credits_array)

    if mean == 0:
        return 0.0

    # Calculate Gini coefficient
    diff_sum = np.sum(np.abs(credits_array[:, None] - credits_array[None, :]))
    gini = diff_sum / (2 * len(credits) ** 2 * mean)

    # Return 1 - gini as fairness index (higher is more fair)
    return 1 - gini


def jain_fairness_index(credits: List[float]) -> float:
    """Calculate Jain's fairness index for credit distribution.

    Jain's fairness index ranges from 1/n (worst case) to 1 (perfect fairness).

    Parameters
    ----------
    credits : List[float]
        List of credit values for agents.

    Returns
    -------
    float
        Jain's fairness index (0.0 to 1.0). Higher is more fair.
    """
    if len(credits) == 0:
        return 0.0

    credits_array = np.array(credits)
    sum_credits = np.sum(credits_array)
    sum_squared = np.sum(credits_array ** 2)

    if sum_squared == 0:
        return 0.0

    n = len(credits)
    jain_index = (sum_credits ** 2) / (n * sum_squared)

    return jain_index


def credit_entropy(credits: Dict[str, float]) -> float:
    """Calculate Shannon entropy of credit distribution.

    Higher entropy indicates more uniform distribution (more fair).

    Parameters
    ----------
    credits : Dict[str, float]
        Dictionary mapping agent IDs to credit values.

    Returns
    -------
    float
        Shannon entropy.
    """
    if not credits:
        return 0.0

    credit_values = np.array(list(credits.values()))
    total = credit_values.sum()

    if total == 0:
        return 0.0

    # Normalize to probability distribution
    probs = credit_values / total

    # Filter out zero probabilities
    probs = probs[probs > 0]

    # Calculate Shannon entropy
    entropy = -np.sum(probs * np.log(probs))

    return float(entropy)


def calculate_all_metrics(
    predictions: List[str],
    gold_answers: List[str],
    static_credits: List[Dict[str, float]],
    dynamic_credits: List[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Calculate all evaluation metrics for multi-hop QA.

    Parameters
    ----------
    predictions : List[str]
        List of predicted answers.
    gold_answers : List[str]
        List of ground truth answers.
    static_credits : List[Dict[str, float]]
        List of static credit allocations for each sample.
    dynamic_credits : List[Dict[str, float]], optional
        List of dynamic credit allocations for each sample.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed metrics.
    """
    metrics = {}

    # QA metrics
    metrics['exact_match_accuracy'] = multi_hop_accuracy(predictions, gold_answers)
    metrics['average_f1'] = average_f1_score(predictions, gold_answers)

    # Static credit fairness
    static_entropies = [credit_entropy(c) for c in static_credits]
    metrics['average_static_entropy'] = np.mean(static_entropies)
    metrics['std_static_entropy'] = np.std(static_entropies)

    # Aggregate static credits across all samples
    all_static_values = []
    for credit_dict in static_credits:
        all_static_values.extend(credit_dict.values())

    if all_static_values:
        metrics['static_fairness_gini'] = fairness_index(all_static_values)
        metrics['static_fairness_jain'] = jain_fairness_index(all_static_values)

    # Dynamic credit fairness (if provided)
    if dynamic_credits:
        dynamic_entropies = [credit_entropy(c) for c in dynamic_credits]
        metrics['average_dynamic_entropy'] = np.mean(dynamic_entropies)
        metrics['std_dynamic_entropy'] = np.std(dynamic_entropies)

        all_dynamic_values = []
        for credit_dict in dynamic_credits:
            all_dynamic_values.extend(credit_dict.values())

        if all_dynamic_values:
            metrics['dynamic_fairness_gini'] = fairness_index(all_dynamic_values)
            metrics['dynamic_fairness_jain'] = jain_fairness_index(all_dynamic_values)

        # Comparison metrics
        entropy_improvement = (
            metrics['average_dynamic_entropy'] - metrics['average_static_entropy']
        )
        metrics['entropy_improvement'] = entropy_improvement

    return metrics


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted report of evaluation metrics.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary containing computed metrics.
    """
    print("\n" + "="*70)
    print("EVALUATION METRICS REPORT")
    print("="*70)

    print("\nQA Performance:")
    print(f"  Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0.0):.3f}")
    print(f"  Average F1 Score:     {metrics.get('average_f1', 0.0):.3f}")

    print("\nStatic Credit Allocation:")
    print(f"  Average Entropy:      {metrics.get('average_static_entropy', 0.0):.3f}")
    print(f"  Std Entropy:          {metrics.get('std_static_entropy', 0.0):.3f}")
    print(f"  Gini Fairness Index:  {metrics.get('static_fairness_gini', 0.0):.3f}")
    print(f"  Jain Fairness Index:  {metrics.get('static_fairness_jain', 0.0):.3f}")

    if 'average_dynamic_entropy' in metrics:
        print("\nDynamic Credit Allocation:")
        print(f"  Average Entropy:      {metrics.get('average_dynamic_entropy', 0.0):.3f}")
        print(f"  Std Entropy:          {metrics.get('std_dynamic_entropy', 0.0):.3f}")
        print(f"  Gini Fairness Index:  {metrics.get('dynamic_fairness_gini', 0.0):.3f}")
        print(f"  Jain Fairness Index:  {metrics.get('dynamic_fairness_jain', 0.0):.3f}")

        print("\nComparison:")
        print(f"  Entropy Improvement:  {metrics.get('entropy_improvement', 0.0):+.3f}")

    print("="*70 + "\n")
