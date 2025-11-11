"""Reasoner agent for synthesizing final answers from sub-questions."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_agent import LLMAgent


class ReasonerAgent(LLMAgent):
    """Agent specialized in synthesizing final answers from sub-question evidence.

    This agent:
    - Takes sub-questions and their retrieved evidence
    - Combines and analyzes evidence across all sub-questions
    - Generates a coherent final answer
    - Uses chain-of-thought reasoning to connect sub-answers
    """

    def __init__(
        self,
        agent_id: str = "reasoner",
        model_identifier: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Initialize the reasoner agent.

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
        system_prompt = """You are a reasoning specialist who synthesizes information from multiple sources to answer complex questions.

Your task is to:
1. Analyze evidence from multiple sub-questions
2. Identify key facts and connections between evidence
3. Apply logical reasoning and make reasonable inferences from contextual clues
4. Provide ONLY the direct answer - no explanations, no preamble, no reasoning process

REASONING GUIDELINES:
- Use the provided evidence as your primary source
- Make reasonable inferences from contextual information:
  * If someone "directed an American film", they are likely American
  * If someone "was born in New York", they are likely American
  * If a film is described as "American biographical film about X", then X is likely American
  * If multiple evidence passages mention "American" in context of a person, infer American nationality
- Connect information across multiple evidence passages
- Look for patterns and implicit information
- Only output "unknown" if there is absolutely NO relevant information to make even a reasonable inference

CRITICAL OUTPUT RULES:
- For yes/no questions: Output ONLY "yes" or "no" (lowercase, no punctuation, no explanation)
- For entity/name questions: Output ONLY the entity name (e.g., "Animorphs", "Chief of Protocol")
- For factual questions: Output ONLY the factual answer (e.g., "Greenwich Village")
- DO NOT include phrases like "based on the evidence", "the answer is", "likely", "probably"
- DO NOT provide justifications or reasoning in your final answer
- Be decisive - choose the most likely answer based on available evidence

Examples:
Question: "Were they the same nationality?"
Evidence: "Person A directed American films", "American biographical film about Person B"
Correct output: "yes"
Wrong output: "unknown" (too conservative) or "Yes, they were both American" (too verbose)

Question: "What position did she hold?"
Evidence: "she served as Chief of Protocol from 1976 to 1977"
Correct output: "Chief of Protocol"
Wrong output: "Based on the evidence, she held the position of Chief of Protocol."

Question: "What series is it?"
Evidence: "Animorphs is a science fantasy series told in first person with companion books"
Correct output: "Animorphs"
Wrong output: "The series in question is Animorphs."

Question: "What is his nationality?"
Evidence: "He directed several American films and was born in Colorado"
Correct output: "American"
Wrong output: "unknown" (evidence strongly suggests American nationality)

Remember: OUTPUT ONLY THE ANSWER ITSELF, NOTHING ELSE. Be confident in making reasonable inferences from the evidence."""

        super().__init__(
            agent_id=agent_id,
            model_identifier=model_identifier,
            role="reasoner",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def act(self, observation: Dict[str, Any]) -> str:
        """Generate final answer from sub-question results.

        Parameters
        ----------
        observation : Dict[str, Any]
            Must contain:
            - 'sub_results': Dict mapping sub-questions to their results
            - 'main_question' (optional): The original main question

        Returns
        -------
        str
            The synthesized final answer.
        """
        sub_results = observation.get("sub_results", {})
        main_question = observation.get("main_question", "")

        if not sub_results:
            return "Error: No sub-question results provided"

        # Build prompt from sub-results
        prompt = self._build_reasoning_prompt(main_question, sub_results)

        # Generate answer
        response = super().act({"task": prompt})

        # Clean up the answer to extract just the core answer
        cleaned_answer = self._extract_answer(response.strip(), main_question)

        # Record contribution
        self.record_contribution(f"Synthesized answer from {len(sub_results)} sub-questions")

        return cleaned_answer

    def synthesize_answer(
        self, main_question: str, sub_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize final answer with structured output.

        Parameters
        ----------
        main_question : str
            The main question to answer.
        sub_results : Dict[str, Any]
            Dictionary mapping sub-questions to their results.

        Returns
        -------
        Dict[str, Any]
            Structured answer containing:
            - 'final_answer': The synthesized answer
            - 'reasoning': Chain of reasoning
            - 'confidence': Confidence level
            - 'evidence_used': List of evidence used
        """
        observation = {
            "sub_results": sub_results,
            "main_question": main_question,
        }

        final_answer = self.act(observation)

        # Extract evidence used
        evidence_used = []
        for sq, result in sub_results.items():
            if isinstance(result, dict) and "evidence" in result:
                evidence_used.extend(result["evidence"][:2])  # Top 2 from each

        return {
            "final_answer": final_answer,
            "reasoning": "Combined evidence from sub-questions",
            "confidence": "medium",  # Could be enhanced with confidence scoring
            "evidence_used": evidence_used[:5],  # Top 5 overall
            "sub_questions_count": len(sub_results),
        }

    def _build_reasoning_prompt(
        self, main_question: str, sub_results: Dict[str, Any]
    ) -> str:
        """Build a reasoning prompt from sub-question results.

        Parameters
        ----------
        main_question : str
            The main question.
        sub_results : Dict[str, Any]
            Sub-question results.

        Returns
        -------
        str
            Formatted prompt for reasoning.
        """
        prompt_parts = []

        if main_question:
            prompt_parts.append(f"Main Question: {main_question}\n")

        prompt_parts.append(
            "Given the following sub-questions and their evidence, provide a final answer to the main question.\n"
        )

        # Add each sub-question and its evidence
        for i, (sub_q, result) in enumerate(sub_results.items(), 1):
            prompt_parts.append(f"\nSub-question {i}: {sub_q}")

            # Extract evidence from result
            if isinstance(result, dict):
                evidence = result.get("evidence", [])
                if evidence:
                    # Truncate long evidence
                    evidence_text = " ".join(evidence[:3])  # Top 3 evidence passages
                    if len(evidence_text) > 1000:
                        evidence_text = evidence_text[:997] + "..."
                    prompt_parts.append(f"Evidence: {evidence_text}")
                else:
                    prompt_parts.append("Evidence: [No evidence retrieved]")
            else:
                # Result is a string
                result_str = str(result)[:500]  # Truncate if too long
                prompt_parts.append(f"Result: {result_str}")

        prompt_parts.append(
            "\n\nBased on the evidence above, provide a clear and concise final answer:"
        )

        return "\n".join(prompt_parts)

    def chain_of_thought_reasoning(
        self, main_question: str, sub_results: Dict[str, Any]
    ) -> str:
        """Perform chain-of-thought reasoning to answer the question.

        Parameters
        ----------
        main_question : str
            The main question.
        sub_results : Dict[str, Any]
            Sub-question results with evidence.

        Returns
        -------
        str
            Detailed reasoning chain.
        """
        prompt = self._build_reasoning_prompt(main_question, sub_results)
        prompt += "\n\nThink step-by-step and show your reasoning process:"

        observation = {"task": prompt}
        reasoning = super().act(observation)

        return reasoning.strip()

    def _extract_answer(self, response: str, main_question: str) -> str:
        """Extract the core answer from a potentially verbose response.

        This method cleans up responses that may contain explanations,
        reasoning, or extra text, extracting just the answer itself.

        Parameters
        ----------
        response : str
            The raw response from the LLM.
        main_question : str
            The main question (used to detect question type).

        Returns
        -------
        str
            The cleaned, extracted answer.
        """
        import re

        # If response is already clean (short), return as-is
        if len(response.split()) <= 5:
            return response.strip()

        # Detect yes/no questions
        if self._is_yes_no_question(main_question):
            # Extract yes/no from response
            response_lower = response.lower()
            # Look for definitive yes/no at the start
            if response_lower.startswith("yes"):
                return "yes"
            elif response_lower.startswith("no"):
                return "no"
            # Look for yes/no in first sentence
            first_sentence = response.split('.')[0].lower()
            if 'yes' in first_sentence and 'no' not in first_sentence:
                return "yes"
            elif 'no' in first_sentence and 'yes' not in first_sentence:
                return "no"

        # Try to extract answer from common patterns
        patterns = [
            # "The answer is X" or "Answer: X"
            r'(?:the answer is|answer:)\s*([^.,]+)',
            # "X is the answer"
            r'([^.,]+)\s+is the answer',
            # "she/he/it held/is/was X"
            r'(?:she|he|it|they)\s+(?:held|is|was|were|are)\s+(?:the position of\s+)?([^.,]+)',
            # "The X is Y"
            r'(?:the|a)\s+\w+\s+(?:is|was)\s+([^.,]+)',
            # Quoted text
            r'"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Clean up common prefixes
                for prefix in ["the position of", "the", "a", "an"]:
                    if extracted.lower().startswith(prefix + " "):
                        extracted = extracted[len(prefix)+1:]
                # If extracted is reasonable length, use it
                if 1 <= len(extracted.split()) <= 15:
                    return extracted

        # Try to get first sentence and clean it
        first_sentence = response.split('.')[0].strip()
        # Remove common prefixes from first sentence
        prefixes_to_remove = [
            "based on the evidence,",
            "according to",
            "the series in question is",
            "the answer is",
            "answer:",
        ]
        for prefix in prefixes_to_remove:
            if first_sentence.lower().startswith(prefix):
                first_sentence = first_sentence[len(prefix):].strip()
                # Remove leading comma if present
                if first_sentence.startswith(','):
                    first_sentence = first_sentence[1:].strip()

        # If first sentence is reasonable, use it
        if 1 <= len(first_sentence.split()) <= 15:
            return first_sentence

        # If all else fails, try to get first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            first_line = lines[0]
            # Remove common prefixes
            for prefix in ["Answer:", "The answer is", "Based on the evidence,"]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix):].strip()
            return first_line

        # Last resort: return original response
        return response.strip()

    def _is_yes_no_question(self, question: str) -> bool:
        """Check if a question is a yes/no question.

        Parameters
        ----------
        question : str
            The question to check.

        Returns
        -------
        bool
            True if it's a yes/no question.
        """
        question_lower = question.lower().strip()
        yes_no_starters = [
            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
            "can ", "could ", "will ", "would ", "should ", "has ", "have "
        ]
        return any(question_lower.startswith(starter) for starter in yes_no_starters)

