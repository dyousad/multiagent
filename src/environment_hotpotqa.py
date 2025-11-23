"""HotpotQA environment for multi-hop question answering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
from environment import Environment, TaskResult


class HotpotQAEnvironment(Environment):
    """Environment for HotpotQA multi-hop question answering tasks.

    HotpotQA requires reasoning over multiple documents to answer questions.
    This environment extends the base Environment class to handle HotpotQA-specific
    data and evaluation.
    """

    def __init__(
        self,
        data_path: str = "data/hotpot_dev_fullwiki_v1.json",
        max_samples: Optional[int] = 100,
        split: str = "validation"
    ):
        """Initialize the HotpotQA environment.

        Parameters
        ----------
        data_path : str
            Path to the HotpotQA JSON file.
        max_samples : Optional[int]
            Maximum number of samples to load. None means load all.
        split : str
            Data split to use (validation, train, test).
        """
        super().__init__(task_description="")
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        self.current_sample_idx: int = 0

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load HotpotQA data from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Select subset if max_samples is specified
        if self.max_samples:
            self.samples = random.sample(data,self.max_samples)
            # self.samples = data[:self.max_samples]
        else:
            self.samples = data

        print(f"Loaded {len(self.samples)} samples from {self.data_path}")

    def get_task(self, idx: int) -> Dict[str, Any]:
        """Get a specific task by index.

        Parameters
        ----------
        idx : int
            Index of the task sample.

        Returns
        -------
        Dict[str, Any]
            Task dictionary with question, context, answer, and supporting facts.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range (total: {len(self.samples)})")

        item = self.samples[idx]

        # Format context from list of [title, sentences] pairs
        context_str = self._format_context(item.get("context", []))

        return {
            "question": item.get("question", ""),
            "context": context_str,
            "raw_context": item.get("context", []),
            "answer": item.get("answer", ""),
            "supporting_facts": item.get("supporting_facts", {}),
            "type": item.get("type", ""),  # comparison or bridge
            "_id": item.get("_id", "")
        }

    def _format_context(self, context: List[List]) -> str:
        """Format context from HotpotQA format to readable string.

        Parameters
        ----------
        context : List[List]
            List of [title, sentences] pairs.

        Returns
        -------
        str
            Formatted context string.
        """
        formatted_parts = []
        for title, sentences in context:
            formatted_parts.append(f"# {title}")
            for sent in sentences:
                if sent:  # Skip empty sentences
                    formatted_parts.append(sent)
            formatted_parts.append("")  # Add blank line between documents

        return "\n".join(formatted_parts)

    def set_current_task(self, idx: int) -> None:
        """Set the current task by index.

        Parameters
        ----------
        idx : int
            Index of the task to set as current.
        """
        task = self.get_task(idx)
        self.task_description = task["question"]
        self.current_sample_idx = idx
        # Store full task info for evaluation
        self.current_task_data = task
        # Reset environment state
        self.agent_responses.clear()
        self.current_round = 0
        self.completed = False

    def evaluate(self, final_responses: Dict[str, str]) -> TaskResult:
        """Evaluate the task completion for HotpotQA.

        Parameters
        ----------
        final_responses : Dict[str, str]
            Final responses from all agents.

        Returns
        -------
        TaskResult
            Result of the task execution with HotpotQA-specific metrics.
        """
        # Combine all responses into final output
        final_output = "\n\n".join([
            f"Agent {agent_id}: {response}"
            for agent_id, response in final_responses.items()
        ])

        # Extract predicted answer (simple heuristic: last agent's response)
        predicted_answer = list(final_responses.values())[-1] if final_responses else ""

        # Get ground truth answer
        ground_truth = self.current_task_data.get("answer", "")

        # Exact match evaluation
        exact_match = self._evaluate_exact_match(predicted_answer, ground_truth)

        # F1 score evaluation
        f1_score = self._evaluate_f1(predicted_answer, ground_truth)

        success = exact_match or f1_score > 0.5

        self.completed = True

        return TaskResult(
            task_description=self.task_description,
            agent_responses=final_responses,
            final_output=final_output,
            success=success,
            metadata={
                "rounds": self.current_round,
                "num_agents": len(final_responses),
                "exact_match": exact_match,
                "f1_score": f1_score,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "question_type": self.current_task_data.get("type", ""),
                "question_id": self.current_task_data.get("_id", ""),
            }
        )

    def _evaluate_exact_match(self, prediction: str, ground_truth: str) -> bool:
        """Evaluate exact match between prediction and ground truth.

        Parameters
        ----------
        prediction : str
            Predicted answer.
        ground_truth : str
            Ground truth answer.

        Returns
        -------
        bool
            True if exact match, False otherwise.
        """
        # Normalize strings: lowercase, strip whitespace
        pred_normalized = prediction.lower().strip()
        truth_normalized = ground_truth.lower().strip()

        return pred_normalized == truth_normalized

    def _evaluate_f1(self, prediction: str, ground_truth: str) -> float:
        """Evaluate F1 score between prediction and ground truth.

        Parameters
        ----------
        prediction : str
            Predicted answer.
        ground_truth : str
            Ground truth answer.

        Returns
        -------
        float
            F1 score.
        """
        # Tokenize by splitting on whitespace
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0

        # Calculate precision and recall
        common_tokens = pred_tokens & truth_tokens

        if len(common_tokens) == 0:
            return 0.0

        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)

        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    def get_num_samples(self) -> int:
        """Get the total number of samples loaded.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.samples)
