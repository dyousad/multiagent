"""Environment for multi-agent task execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a multi-agent task execution."""

    task_description: str
    agent_responses: Dict[str, str]  # agent_id -> response
    final_output: str
    success: bool
    metadata: Dict[str, Any]


class Environment:
    """Environment that coordinates multi-agent task execution.

    The environment:
    - Manages the task state
    - Coordinates agent interactions
    - Evaluates task completion
    - Provides observations to agents
    """

    def __init__(self, task_description: str):
        """Initialize the environment with a task.

        Parameters
        ----------
        task_description : str
            Description of the task to be solved.
        """
        self.task_description = task_description
        self.agent_responses: Dict[str, List[str]] = {}
        self.current_round: int = 0
        self.max_rounds: int = 5
        self.completed: bool = False

    def get_observation(self, agent_id: str, previous_responses: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get the current observation for an agent.

        Parameters
        ----------
        agent_id : str
            ID of the agent requesting the observation.
        previous_responses : Optional[List[str]]
            Responses from other agents in the current round.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with task info and context.
        """
        obs = {
            "task": self.task_description,
            "round": self.current_round,
            "agent_id": agent_id,
            "previous_responses": previous_responses or [],
        }

        # Add history of this agent's previous responses
        if agent_id in self.agent_responses:
            obs["agent_history"] = self.agent_responses[agent_id]

        return obs

    def record_response(self, agent_id: str, response: str) -> None:
        """Record an agent's response.

        Parameters
        ----------
        agent_id : str
            ID of the agent.
        response : str
            The agent's response.
        """
        if agent_id not in self.agent_responses:
            self.agent_responses[agent_id] = []
        self.agent_responses[agent_id].append(response)

    def next_round(self) -> bool:
        """Advance to the next round.

        Returns
        -------
        bool
            True if task should continue, False if max rounds reached.
        """
        self.current_round += 1
        return self.current_round < self.max_rounds

    def evaluate(self, final_responses: Dict[str, str]) -> TaskResult:
        """Evaluate the task completion.

        Parameters
        ----------
        final_responses : Dict[str, str]
            Final responses from all agents.

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        # Combine all responses into final output
        final_output = "\n\n".join([
            f"Agent {agent_id}: {response}"
            for agent_id, response in final_responses.items()
        ])

        # Simple success criterion: all agents provided responses
        success = len(final_responses) > 0 and all(
            len(resp.strip()) > 0 for resp in final_responses.values()
        )

        self.completed = True

        return TaskResult(
            task_description=self.task_description,
            agent_responses=final_responses,
            final_output=final_output,
            success=success,
            metadata={
                "rounds": self.current_round,
                "num_agents": len(final_responses),
            }
        )

    def reset(self, task_description: Optional[str] = None) -> None:
        """Reset the environment for a new task.

        Parameters
        ----------
        task_description : Optional[str]
            New task description, or None to keep current.
        """
        if task_description:
            self.task_description = task_description
        self.agent_responses.clear()
        self.current_round = 0
        self.completed = False
