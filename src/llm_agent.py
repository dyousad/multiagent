"""LLM-powered agent implementation."""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, Optional

# Add parent directory to path to import api_models
# sys.path.insert(0, os.path.expanduser("~/experiment"))

from api_models import APILanguageModel, get_api_model
from agent import Agent


class LLMAgent(Agent):
    """An agent powered by a language model API.

    This agent uses an external LLM (e.g., Deepseek, Qwen) to generate
    responses based on observations and conversation context.
    """

    def __init__(
        self,
        agent_id: str,
        model_identifier: str = "deepseek-ai/DeepSeek-V3",
        role: str = "assistant",
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize an LLM-powered agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        model_identifier : str
            Model identifier for API (e.g., "deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-7B-Instruct").
        role : str
            The role or persona of this agent.
        system_prompt : Optional[str]
            System prompt to guide the agent's behavior.
        max_tokens : int
            Maximum tokens for LLM generation.
        temperature : float
            Sampling temperature for LLM.
        """
        super().__init__(agent_id=agent_id, role=role)
        self.model_identifier = model_identifier
        self.llm: APILanguageModel = get_api_model(model_identifier)
        self.system_prompt = system_prompt or f"You are {role} agent {agent_id}."
        self.max_tokens = max_tokens
        self.temperature = temperature

    def act(self, observation: Dict[str, Any]) -> str:
        """Generate an action using the LLM based on observation.

        Parameters
        ----------
        observation : Dict[str, Any]
            Current task state, including 'task', 'context', etc.

        Returns
        -------
        str
            The agent's generated response.
        """
        # Build prompt from system prompt, observation, and conversation history
        prompt = self._build_prompt(observation)

        # Call LLM
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            print("\nthe usage of token is")
            print(self.llm.get_last_usage())
            # Record this as a contribution
            self.record_contribution(response)
            return response
        except Exception as e:
            error_msg = f"[Agent {self.agent_id}] LLM generation failed: {e}"
            print(error_msg)
            return error_msg

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Construct the prompt for the LLM.

        Parameters
        ----------
        observation : Dict[str, Any]
            Task observation with keys like 'task', 'context', 'previous_responses'.

        Returns
        -------
        str
            The complete prompt string.
        """
        lines = [self.system_prompt, ""]

        # Add task description
        if "task" in observation:
            lines.append(f"Task: {observation['task']}")
            lines.append("")

        # Add additional context
        if "context" in observation:
            lines.append(f"Context: {observation['context']}")
            lines.append("")

        # Add previous agent responses
        if "previous_responses" in observation:
            lines.append("Previous agent responses:")
            for resp in observation["previous_responses"]:
                lines.append(f"  - {resp}")
            lines.append("")

        # Add conversation history
        if self.message_history:
            lines.append("Conversation history:")
            for msg in self.message_history:
                sender = "You" if msg.sender_id == self.agent_id else msg.sender_id
                receiver = msg.receiver_id or "all"
                lines.append(f"  {sender} -> {receiver}: {msg.content}")
            lines.append("")

        lines.append("Your response:")
        return "\n".join(lines)
