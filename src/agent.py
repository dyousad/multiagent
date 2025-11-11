"""Agent base class for multi-agent collaboration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Message:
    """A message exchanged between agents."""

    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base class for an agent in the multi-agent system.

    Each agent can:
    - Act on observations from the environment
    - Communicate with other agents
    - Receive messages from other agents
    - Track its contribution to task completion
    """

    def __init__(self, agent_id: str, role: str = "assistant"):
        """Initialize an agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        role : str
            The role or persona of this agent (e.g., "coder", "reviewer", "planner").
        """
        self.agent_id = agent_id
        self.role = role
        self.message_history: List[Message] = []
        self.contribution_history: List[str] = []
        self.reward: float = 0.0

    def act(self, observation: Dict[str, Any]) -> str:
        """Generate an action based on the current observation.

        Parameters
        ----------
        observation : Dict[str, Any]
            Current state of the environment or task context.

        Returns
        -------
        str
            The agent's response or action.
        """
        raise NotImplementedError("Subclasses must implement act()")

    def communicate(self, message: str, receiver_id: Optional[str] = None) -> Message:
        """Send a message to another agent or broadcast to all.

        Parameters
        ----------
        message : str
            The content to communicate.
        receiver_id : Optional[str]
            ID of the receiving agent, or None to broadcast.

        Returns
        -------
        Message
            The message object that was sent.
        """
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=message
        )
        self.message_history.append(msg)
        return msg

    def receive(self, message: Message) -> None:
        """Receive and process a message from another agent.

        Parameters
        ----------
        message : Message
            The incoming message.
        """
        if message.receiver_id is None or message.receiver_id == self.agent_id:
            self.message_history.append(message)

    def record_contribution(self, contribution: str) -> None:
        """Record a contribution made by this agent.

        Parameters
        ----------
        contribution : str
            Description of the contribution.
        """
        self.contribution_history.append(contribution)

    def set_reward(self, reward: float) -> None:
        """Set the reward received by this agent.

        Parameters
        ----------
        reward : float
            The Shapley value or other reward metric.
        """
        self.reward = reward

    def get_context(self) -> str:
        """Get the agent's conversation context as a string.

        Returns
        -------
        str
            Formatted conversation history.
        """
        context_lines = [f"Agent {self.agent_id} (Role: {self.role})"]
        for msg in self.message_history:
            if msg.sender_id == self.agent_id:
                context_lines.append(f"  You -> {msg.receiver_id or 'all'}: {msg.content}")
            else:
                context_lines.append(f"  {msg.sender_id} -> you: {msg.content}")
        return "\n".join(context_lines)
