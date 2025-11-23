"""Decomposer agent for breaking down complex questions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm_agent import LLMAgent


class DecomposerAgent(LLMAgent):
    """Agent specialized in decomposing complex multi-hop questions.

    This agent analyzes a complex question and breaks it down into
    simpler sub-questions that can be answered independently or sequentially.
    """

    def __init__(
        self,
        agent_id: str = "decomposer",
        model_identifier: str = "deepseek-ai/DeepSeek-V3",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize the decomposer agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        model_identifier : str
            Model identifier for API.
        max_tokens : int
            Maximum tokens for LLM generation.
        temperature : float
            Sampling temperature for LLM.
        """
        system_prompt = """
You are a question decomposition specialist. Your task is to analyze complex multi-hop questions and break them down into simpler sub-questions.

When given a question:
1. Identify the key reasoning steps needed
2. Break down the question into 2-3 sub-questions
3. Order the sub-questions logically (from foundational to final)
4. Each sub-question should be answerable independently

CRITICAL RULES:
- AVOID pronouns (this, that, it, he, she, they) in sub-questions
- Use explicit entity names or descriptive phrases instead of pronouns
- Make each sub-question self-contained and searchable
- For dependent questions, use placeholders like "the X" instead of pronouns

Format your response as a numbered list:
1. [First sub-question]
2. [Second sub-question]
3. [Third sub-question]

Good Example 1:
Question: "Which movie has a higher budget, the one directed by Christopher Nolan in 2010 or the one directed by James Cameron in 2009?"
Sub-questions:
1. Which movie was directed by Christopher Nolan in 2010?
2. What was the budget of the Christopher Nolan 2010 movie?
3. Which movie was directed by James Cameron in 2009?
4. What was the budget of the James Cameron 2009 movie?

Good Example 2:
Question: "What government position was held by the woman who portrayed Corliss Archer in Kiss and Tell?"
Sub-questions:
1. Who portrayed Corliss Archer in the film Kiss and Tell?
2. What government position did Shirley Temple hold?
Note: Use the actress name from question 1, or if unknown, use "the actress who portrayed Corliss Archer"

Good Example 3:
Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
Sub-questions:
1. What is Scott Derrickson's nationality?
2. What is Ed Wood's nationality?

Bad Example (avoid this):
Question: "What position was held by the woman who portrayed Corliss Archer?"
Sub-questions:
1. Who portrayed Corliss Archer in the film Kiss and Tell?
2. What government position did this woman hold?  ← BAD: uses "this woman" instead of explicit name"""

        super().__init__(
            agent_id=agent_id,
            model_identifier=model_identifier,
            role="decomposer",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def act(self, observation: Dict[str, Any]) -> str:
        """Decompose the question in the observation.

        Parameters
        ----------
        observation : Dict[str, Any]
            Must contain 'task' or 'question' key with the complex question.

        Returns
        -------
        str
            The decomposition result with sub-questions.
        """
        # Call parent's act method to get LLM response
        response = super().act(observation)
        return response

    def parse_sub_questions(self, decomposition_text: str) -> List[str]:
        """Parse sub-questions from the decomposition text.

        Parameters
        ----------
        decomposition_text : str
            The text output from the decomposer containing sub-questions.

        Returns
        -------
        List[str]
            List of sub-questions extracted from the text.
        """
        sub_questions = []

        # Pattern to match numbered items (e.g., "1. Question here")
        pattern = r'^\s*\d+[\.\)]\s*(.+)$'

        for line in decomposition_text.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                question = match.group(1).strip()
                if question:
                    sub_questions.append(question)

        return sub_questions

    def decompose_question(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Decompose a question into sub-questions.

        Parameters
        ----------
        question : str
            The complex question to decompose.
        context : Optional[str]
            Additional context for the question.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'original_question': The original question
            - 'decomposition_text': Raw decomposition output
            - 'sub_questions': List of parsed sub-questions
        """
        # Build observation
        observation = {
            "task": question,
        }
        if context:
            observation["context"] = context

        # Get decomposition
        decomposition_text = self.act(observation)

        # Parse sub-questions
        sub_questions = self.parse_sub_questions(decomposition_text)

        return {
            "original_question": question,
            "decomposition_text": decomposition_text,
            "sub_questions": sub_questions,
        }
