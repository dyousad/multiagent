"""Reward management for multi-agent systems."""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path

from shapley import compute_shapley


class RewardManager:
    """Manages reward allocation to agents based on Shapley values.

    The reward manager:
    - Computes Shapley values for agent contributions
    - Allocates rewards to agents
    - Logs reward history
    - Supports different reward allocation strategies
    - Supports dynamic credit reallocation based on test-time performance
    """

    def __init__(
        self,
        base_reward: float = 1.0,
        log_file: Optional[Path] = None,
        role_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the reward manager.

        Parameters
        ----------
        base_reward : float
            Base reward to be distributed among agents.
        log_file : Optional[Path]
            Path to log file for reward history.
        role_weights : Optional[Dict[str, float]]
            Weights for different agent roles in credit allocation.
        """
        self.base_reward = base_reward
        self.log_file = log_file
        self.reward_history: List[Dict[str, Any]] = []
        self.credit: Dict[str, float] = {}  # Dynamic credit tracking
        self.role_weights = role_weights or {}  # Role-based weights

    def allocate_shapley_rewards(
        self,
        agents: List[Any],  # List[Agent]
        contributions: Dict[str, float],
        value_function: Optional[Callable[[Set[str]], float]] = None,
        use_monte_carlo: bool = True,
        num_samples: int = 1000
    ) -> Dict[str, float]:
        """Allocate rewards based on Shapley values.

        Parameters
        ----------
        agents : List[Agent]
            List of agents to reward.
        contributions : Dict[str, float]
            Individual contribution scores for each agent.
        value_function : Optional[Callable[[Set[str]], float]]
            Custom value function for coalitions.
        use_monte_carlo : bool
            Whether to use Monte Carlo approximation.
        num_samples : int
            Number of samples for Monte Carlo.

        Returns
        -------
        Dict[str, float]
            Reward allocated to each agent.
        """
        agent_ids = [agent.agent_id for agent in agents]

        # Compute Shapley values
        shapley_values = compute_shapley(
            contributions=contributions,
            agents=agent_ids,
            value_function=value_function,
            use_monte_carlo=use_monte_carlo,
            num_samples=num_samples
        )

        # Allocate rewards: reward = shapley_value * base_reward
        rewards = {
            agent_id: shapley_value * self.base_reward
            for agent_id, shapley_value in shapley_values.items()
        }

        # Update agent rewards
        for agent in agents:
            agent.set_reward(rewards[agent.agent_id])

        # Log rewards
        self._log_rewards(rewards, shapley_values, contributions)

        return rewards

    def allocate_uniform_rewards(self, agents: List[Any]) -> Dict[str, float]:
        """Allocate uniform rewards to all agents (baseline).

        Parameters
        ----------
        agents : List[Agent]
            List of agents to reward.

        Returns
        -------
        Dict[str, float]
            Uniform reward allocated to each agent.
        """
        n = len(agents)
        uniform_reward = self.base_reward / n

        rewards = {agent.agent_id: uniform_reward for agent in agents}

        # Update agent rewards
        for agent in agents:
            agent.set_reward(uniform_reward)

        # Log rewards
        self._log_rewards(rewards, None, None, method="uniform")

        return rewards

    def allocate_proportional_rewards(
        self,
        agents: List[Any],
        contributions: Dict[str, float]
    ) -> Dict[str, float]:
        """Allocate rewards proportional to contributions (baseline).

        Parameters
        ----------
        agents : List[Agent]
            List of agents to reward.
        contributions : Dict[str, float]
            Individual contribution scores.

        Returns
        -------
        Dict[str, float]
            Proportional reward allocated to each agent.
        """
        total_contribution = sum(contributions.values())

        if total_contribution == 0:
            # Fall back to uniform allocation
            return self.allocate_uniform_rewards(agents)

        rewards = {
            agent_id: (contrib / total_contribution) * self.base_reward
            for agent_id, contrib in contributions.items()
        }

        # Update agent rewards
        for agent in agents:
            agent.set_reward(rewards[agent.agent_id])

        # Log rewards
        self._log_rewards(rewards, None, contributions, method="proportional")

        return rewards

    def _log_rewards(
        self,
        rewards: Dict[str, float],
        shapley_values: Optional[Dict[str, float]],
        contributions: Optional[Dict[str, float]],
        method: str = "shapley"
    ) -> None:
        """Log reward allocation.

        Parameters
        ----------
        rewards : Dict[str, float]
            Rewards allocated to each agent.
        shapley_values : Optional[Dict[str, float]]
            Shapley values (if applicable).
        contributions : Optional[Dict[str, float]]
            Individual contributions (if applicable).
        method : str
            Reward allocation method.
        """
        log_entry = {
            "method": method,
            "rewards": rewards,
            "shapley_values": shapley_values,
            "contributions": contributions,
        }

        self.reward_history.append(log_entry)

        if self.log_file:
            self._save_log()

    def _save_log(self) -> None:
        """Save reward history to log file."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.reward_history, f, indent=2)

    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward distribution.

        Returns
        -------
        Dict[str, Any]
            Statistics including mean, variance, and fairness metrics.
        """
        if not self.reward_history:
            return {}

        # Calculate statistics from last reward allocation
        last_rewards = self.reward_history[-1]["rewards"]
        reward_values = list(last_rewards.values())

        mean_reward = sum(reward_values) / len(reward_values)
        variance = sum((r - mean_reward) ** 2 for r in reward_values) / len(reward_values)

        # Fairness index (Jain's fairness index)
        sum_rewards = sum(reward_values)
        sum_squared = sum(r ** 2 for r in reward_values)
        n = len(reward_values)
        fairness_index = (sum_rewards ** 2) / (n * sum_squared) if sum_squared > 0 else 0

        return {
            "mean_reward": mean_reward,
            "variance": variance,
            "std_deviation": variance ** 0.5,
            "fairness_index": fairness_index,
            "min_reward": min(reward_values),
            "max_reward": max(reward_values),
        }

    def update_credits_dynamic(
        self,
        agent_outputs: Dict[str, str],
        final_answer: str,
        ground_truth: str,
        evaluate_fn: Optional[Callable[[str, str], float]] = None,
        agent_roles: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Dynamically update agent credits based on counterfactual analysis.

        This implements test-time credit reallocation by computing marginal
        contributions of each agent to the final answer quality, with optional
        role-based weighting.

        Parameters
        ----------
        agent_outputs : Dict[str, str]
            Dictionary mapping agent_id to their output/contribution.
        final_answer : str
            The final answer produced by the multi-agent system.
        ground_truth : str
            The ground truth answer for evaluation.
        evaluate_fn : Optional[Callable[[str, str], float]]
            Function to evaluate answer quality (prediction, ground_truth) -> score.
            If None, uses exact match.
        agent_roles : Optional[Dict[str, str]]
            Dictionary mapping agent_id to their role for weight application.

        Returns
        -------
        Dict[str, float]
            Updated credit scores for each agent.
        """
        # Initialize credits if not already done
        for agent_id in agent_outputs.keys():
            if agent_id not in self.credit:
                self.credit[agent_id] = 0.0

        # Default evaluation function: exact match
        if evaluate_fn is None:
            def exact_match(pred: str, truth: str) -> float:
                return 1.0 if pred.strip().lower() == truth.strip().lower() else 0.0
            evaluate_fn = exact_match

        # Baseline score with all agents
        baseline_score = evaluate_fn(final_answer, ground_truth)

        # Compute marginal contribution of each agent via counterfactual
        marginal_contributions = {}
        for agent_id in agent_outputs.keys():
            # Create counterfactual: what if this agent didn't contribute?
            counterfactual_outputs = {
                aid: output for aid, output in agent_outputs.items()
                if aid != agent_id
            }

            # Approximate counterfactual answer (simple heuristic: combine remaining outputs)
            if counterfactual_outputs:
                counterfactual_answer = " ".join(counterfactual_outputs.values())
            else:
                counterfactual_answer = ""

            counterfactual_score = evaluate_fn(counterfactual_answer, ground_truth)

            # Marginal contribution: how much did this agent improve the score?
            delta = baseline_score - counterfactual_score
            marginal_contributions[agent_id] = max(delta, 0.0)

        # Apply role weights if provided
        if agent_roles and self.role_weights:
            weighted_contributions = {}
            for agent_id, contribution in marginal_contributions.items():
                agent_role = agent_roles.get(agent_id, None)
                weight = self.role_weights.get(agent_role, 1.0) if agent_role else 1.0
                weighted_contributions[agent_id] = contribution * weight
            marginal_contributions = weighted_contributions

        # Update credit (accumulate positive contributions)
        for agent_id, contribution in marginal_contributions.items():
            self.credit[agent_id] += contribution

        return self.credit.copy()

    def get_credit_entropy(self, credits: Optional[Dict[str, float]] = None) -> float:
        """Calculate the entropy of credit distribution.

        Higher entropy indicates more uniform credit distribution (fairer).
        Lower entropy indicates more concentrated credit (less fair).

        Parameters
        ----------
        credits : Optional[Dict[str, float]]
            Credit dictionary. If None, uses self.credit.

        Returns
        -------
        float
            Entropy of the credit distribution.
        """
        if credits is None:
            credits = self.credit

        if not credits:
            return 0.0

        credit_values = np.array(list(credits.values()))

        # Normalize to probability distribution
        total_credit = credit_values.sum()
        if total_credit == 0:
            return 0.0

        probs = credit_values / total_credit

        # Compute Shannon entropy
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def reset_credits(self) -> None:
        """Reset all agent credits to zero."""
        self.credit.clear()

