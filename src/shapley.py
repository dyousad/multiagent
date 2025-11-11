"""Shapley value computation for agent credit assignment.

This module implements Monte Carlo approximation of Shapley values to assign
credit to agents based on their marginal contributions to task success.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Set
from itertools import combinations


def compute_shapley_exact(
    agents: List[str],
    value_function: Callable[[Set[str]], float]
) -> Dict[str, float]:
    """Compute exact Shapley values for all agents.

    This is only practical for small numbers of agents (< 10) due to
    exponential complexity O(2^n).

    Parameters
    ----------
    agents : List[str]
        List of agent IDs.
    value_function : Callable[[Set[str]], float]
        Function that returns the value (performance) of a coalition of agents.
        Takes a set of agent IDs and returns a float score.

    Returns
    -------
    Dict[str, float]
        Shapley value for each agent.
    """
    n = len(agents)
    shapley_values = {agent: 0.0 for agent in agents}

    # For each agent i
    for i, agent in enumerate(agents):
        # For each possible coalition size k
        for k in range(n):
            # Get all coalitions of size k not containing agent i
            other_agents = [a for j, a in enumerate(agents) if j != i]
            for coalition in combinations(other_agents, k):
                coalition_set = set(coalition)
                coalition_with_i = coalition_set | {agent}

                # Marginal contribution: V(S ∪ {i}) - V(S)
                marginal = value_function(coalition_with_i) - value_function(coalition_set)

                # Weight: |S|! * (n - |S| - 1)! / n!
                weight = 1.0 / (n * comb(n - 1, k))

                shapley_values[agent] += weight * marginal

    return shapley_values


def compute_shapley_monte_carlo(
    agents: List[str],
    value_function: Callable[[Set[str]], float],
    num_samples: int = 1000
) -> Dict[str, float]:
    """Compute Shapley values using Monte Carlo approximation.

    This is the recommended method for larger numbers of agents.
    Complexity: O(num_samples * n * evaluation_time)

    Parameters
    ----------
    agents : List[str]
        List of agent IDs.
    value_function : Callable[[Set[str]], float]
        Function that returns the value (performance) of a coalition of agents.
    num_samples : int
        Number of random permutations to sample.

    Returns
    -------
    Dict[str, float]
        Approximate Shapley value for each agent.
    """
    n = len(agents)
    shapley_values = {agent: 0.0 for agent in agents}

    for _ in range(num_samples):
        # Random permutation of agents
        perm = agents.copy()
        random.shuffle(perm)

        # For each agent in the permutation
        coalition = set()
        for agent in perm:
            # Marginal contribution: V(S ∪ {agent}) - V(S)
            value_before = value_function(coalition)
            coalition.add(agent)
            value_after = value_function(coalition)
            marginal = value_after - value_before

            shapley_values[agent] += marginal

    # Average over all samples
    for agent in shapley_values:
        shapley_values[agent] /= num_samples

    return shapley_values


def normalize_shapley_values(shapley_values: Dict[str, float]) -> Dict[str, float]:
    """Normalize Shapley values to sum to 1.

    Parameters
    ----------
    shapley_values : Dict[str, float]
        Raw Shapley values.

    Returns
    -------
    Dict[str, float]
        Normalized Shapley values.
    """
    total = sum(shapley_values.values())
    if total == 0:
        # If all values are zero, distribute equally
        n = len(shapley_values)
        return {agent: 1.0 / n for agent in shapley_values}

    return {agent: value / total for agent, value in shapley_values.items()}


def compute_shapley(
    contributions: Dict[str, float],
    agents: List[str],
    value_function: Optional[Callable[[Set[str]], float]] = None,
    use_monte_carlo: bool = True,
    num_samples: int = 1000
) -> Dict[str, float]:
    """High-level function to compute Shapley values.

    Parameters
    ----------
    contributions : Dict[str, float]
        Individual contribution scores for each agent (used if value_function is None).
    agents : List[str]
        List of agent IDs.
    value_function : Optional[Callable[[Set[str]], float]]
        Custom value function for coalitions. If None, uses simple additive model.
    use_monte_carlo : bool
        Whether to use Monte Carlo approximation (recommended for n > 8).
    num_samples : int
        Number of samples for Monte Carlo approximation.

    Returns
    -------
    Dict[str, float]
        Normalized Shapley values for each agent.
    """
    # Default value function: sum of individual contributions
    if value_function is None:
        def default_value_function(coalition: Set[str]) -> float:
            return sum(contributions.get(agent, 0.0) for agent in coalition)
        value_function = default_value_function

    # Choose computation method
    if use_monte_carlo or len(agents) > 8:
        shapley_values = compute_shapley_monte_carlo(agents, value_function, num_samples)
    else:
        shapley_values = compute_shapley_exact(agents, value_function)

    # Normalize to sum to 1
    return normalize_shapley_values(shapley_values)


def comb(n: int, k: int) -> int:
    """Compute binomial coefficient n choose k.

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.

    Returns
    -------
    int
        Binomial coefficient.
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1

    # Use multiplicative formula to avoid overflow
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result
