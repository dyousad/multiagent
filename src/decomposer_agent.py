"""Decomposer agent for breaking down complex questions."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

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
                You are a question decomposition specialist. Your task is to analyze complex multi-hop questions and break them down into
                simpler sub-questions. Return your response strictly as a JSON string (NOT a Python dict), with the following schema:

                {
                    "original_question": "<the original question string>",
                    "sub_questions": ["<subq1>", "<subq2>", ...],
                    "keywords": ["<kw1>", "<kw2>", ...],
                    "notes": "<optional freeform notes>"
                }

                Guidelines for content:
                - Provide 2-5 atomic sub-questions ordered logically (foundational -> final).
                - Keywords should be concise search phrases for each corresponding sub-question (same length as sub_questions list).
                - Putting names and special nouns in front of Keywords
                - Avoid pronouns (this, that, it, he, she, they) — use explicit entity names or descriptive phrases.
                - For dependent questions refer to earlier answers using identifiers like "ANSWER_1", "ANSWER_2".

                CRITICAL: Output MUST be valid JSON string. If you cannot produce JSON, return a minimal JSON with 'notes' describing the issue.
                The Output should not contain nothing other than pure JSON string, especially quote or indication

                Example:
                {
                    "original_question": "Which movie has a higher budget, the one directed by Christopher Nolan in 2010 or the one directed by James Cameron in 2009?",
                    "sub_questions": [
                        "Which movie was directed by Christopher Nolan in 2010?",
                        "What was the budget of ANSWER_1?",
                        "Which movie was directed by James Cameron in 2009?",
                        "What was the budget of ANSWER_3?"
                    ],
                    "keywords": [
                        "Christopher Nolan movie 2010",
                        "ANSWER_1 budget",
                        "James Cameron movie 2009",
                        "ANSWER_3 budget"
                    ],
                    "notes": ""
                }
        """

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
        # Try to parse JSON first (new preferred format)
        try:
            payload = json.loads(decomposition_text.strip())
            if isinstance(payload, dict):
                sub_questions = payload.get("sub_questions", [])
                if isinstance(sub_questions, list):
                    return [str(s).strip() for s in sub_questions if s]
        except Exception:
            # fall through to legacy parsing
            pass

        # Legacy fallback: numbered list parsing
        sub_questions = []
        pattern = r'^\s*\d+[\\.\)]\s*(.+)$'
        for line in decomposition_text.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                question = match.group(1).strip()
                if question:
                    sub_questions.append(question)
        return sub_questions

    def parse_decomposition_json(self, decomposition_text: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Parse decomposition JSON and return (sub_questions, keywords, raw_payload).

        Returns empty lists and empty dict if parsing fails.
        """
        try:
            payload = json.loads(decomposition_text.strip())
            if isinstance(payload, dict):
                sub_questions = payload.get("sub_questions", []) or []
                keywords = payload.get("keywords", []) or []
                # ensure lists of strings
                sub_questions = [str(s).strip() for s in sub_questions if s]
                keywords = [str(k).strip() for k in keywords if k]
                return sub_questions, keywords, payload
        except Exception:
            pass
        return [], [], {}

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

        # Get decomposition (LLM response)
        decomposition_text = self.act(observation)
        decomposition_text = decomposition_text.replace("`","")
        if decomposition_text.startswith("json"):
            decomposition_text =decomposition_text.replace("json","",1)
        # Try JSON parsing first
        sub_questions, keywords, raw_payload = self.parse_decomposition_json(decomposition_text)
        if sub_questions:
            return {
                "original_question": question,
                "decomposition_text": decomposition_text,
                "sub_questions": sub_questions,
                "keywords": keywords,
                "raw_payload": raw_payload,
            }

        # Fallback to legacy numbered-list parsing
        sub_questions = self.parse_sub_questions(decomposition_text)
        return {
            "original_question": question,
            "decomposition_text": decomposition_text,
            "sub_questions": sub_questions,
            "keywords": [],
            "raw_payload": {},
        }
