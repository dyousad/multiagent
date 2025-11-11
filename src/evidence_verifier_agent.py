"""Improved evidence verifier agent using spaCy entity overlap."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_agent import LLMAgent

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Install with: pip install spacy")


class EvidenceVerifierAgent(LLMAgent):
    """Agent specialized in verifying evidence quality using entity overlap.

    This agent:
    - Verifies whether retrieved evidence mentions required entities
    - Uses spaCy NER to compute entity overlap
    - Falls back to keyword matching if spaCy is unavailable
    - Can trigger rewriting of sub-questions if verification fails
    """

    def __init__(
        self,
        agent_id: str = "verifier",
        model_identifier: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.7,
        min_entity_overlap: float = 0.5,
        use_spacy: bool = True,
    ):
        """Initialize the evidence verifier agent.

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
        min_entity_overlap : float
            Minimum entity overlap ratio required for verification.
        use_spacy : bool
            Whether to use spaCy for entity extraction.
        """
        system_prompt = """You are an evidence verification specialist. Your task is to analyze whether retrieved evidence is relevant to a question.

When given a question and evidence passages:
1. Identify key entities, facts, and keywords in the question
2. Check if these appear in the evidence
3. If evidence seems insufficient or irrelevant, rewrite the question to improve retrieval

When rewriting questions:
- Make them more specific and concrete
- Add context or clarifying details
- Rephrase to target the missing information
- Keep the core intent of the original question"""

        super().__init__(
            agent_id=agent_id,
            model_identifier=model_identifier,
            role="evidence_verifier",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self.min_entity_overlap = min_entity_overlap
        self.use_spacy = use_spacy and SPACY_AVAILABLE

    def act(self, observation: Dict[str, Any]) -> str:
        """Verify evidence quality and suggest improvements.

        Parameters
        ----------
        observation : Dict[str, Any]
            Must contain:
            - 'sub_question' or 'question': The question
            - 'evidence': List of evidence passages

        Returns
        -------
        str
            Verification result and possibly revised question.
        """
        # Extract question and evidence
        question = observation.get("sub_question") or observation.get("question", "")
        evidence = observation.get("evidence", [])

        if not question:
            return "Error: No question provided"

        if not evidence:
            return self._generate_revised_question(question, "No evidence retrieved")

        # Verify evidence
        result = self.verify_evidence(question, evidence)

        # Record contribution
        status = "verified" if result["verified"] else "needs revision"
        self.record_contribution(f"Evidence {status} for question")

        # Format output
        if result["verified"]:
            max_score = max(result.get("overlap_scores", [0]))
            return f"Evidence verified successfully. Max entity overlap: {max_score:.2f}"
        else:
            max_score = max(result.get("overlap_scores", [0]))
            return (
                f"Evidence verification failed. "
                f"Max entity overlap: {max_score:.2f} (threshold: {self.min_entity_overlap:.2f})\n"
                f"Revised question: {result.get('revised_question', question)}"
            )

    def verify_evidence(
        self, question: str, evidence: List[str]
    ) -> Dict[str, Any]:
        """Verify if evidence is relevant to the question using entity overlap.

        Parameters
        ----------
        question : str
            The question.
        evidence : List[str]
            List of evidence passages.

        Returns
        -------
        Dict[str, Any]
            Verification result containing:
            - 'verified': bool
            - 'overlap_scores': List[float]
            - 'revised_question': str (if not verified)
        """
        if self.use_spacy:
            # Use spaCy for entity-based verification
            return self._verify_with_spacy(question, evidence)
        else:
            # Fallback to keyword-based verification
            return self._verify_with_keywords(question, evidence)

    def _verify_with_spacy(
        self, question: str, evidence: List[str]
    ) -> Dict[str, Any]:
        """Verify using spaCy entity extraction.

        Parameters
        ----------
        question : str
            The question.
        evidence : List[str]
            Evidence passages.

        Returns
        -------
        Dict[str, Any]
            Verification result.
        """
        # Extract entities from question
        q_doc = nlp(question)
        q_entities = {ent.text.lower() for ent in q_doc.ents}

        # If no entities in question, fall back to keywords
        if not q_entities:
            return self._verify_with_keywords(question, evidence)

        # Compute overlap for each evidence passage
        overlap_scores = []
        for e in evidence:
            e_doc = nlp(e)
            e_entities = {ent.text.lower() for ent in e_doc.ents}

            if q_entities:
                overlap = len(q_entities & e_entities) / len(q_entities)
            else:
                overlap = 0.0

            overlap_scores.append(overlap)

        # Verification passes if any evidence has sufficient overlap
        max_overlap = max(overlap_scores) if overlap_scores else 0.0
        verified = max_overlap >= self.min_entity_overlap

        result = {
            "verified": verified,
            "overlap_scores": overlap_scores,
            "max_overlap": max_overlap,
        }

        # Generate revised question if not verified
        if not verified:
            revised = self._generate_revised_question(
                question, f"Entity overlap {max_overlap:.2f} below threshold {self.min_entity_overlap:.2f}"
            )
            result["revised_question"] = revised

        return result

    def _verify_with_keywords(
        self, question: str, evidence: List[str]
    ) -> Dict[str, Any]:
        """Fallback verification using keyword matching.

        Parameters
        ----------
        question : str
            The question.
        evidence : List[str]
            Evidence passages.

        Returns
        -------
        Dict[str, Any]
            Verification result.
        """
        # Extract keywords (capitalized words and longer words)
        keywords = self._extract_keywords(question)

        # Check keyword presence in evidence
        overlap_scores = []
        for e in evidence:
            e_lower = e.lower()
            if keywords:
                matches = sum(1 for kw in keywords if kw.lower() in e_lower)
                overlap = matches / len(keywords)
            else:
                overlap = 0.0
            overlap_scores.append(overlap)

        max_overlap = max(overlap_scores) if overlap_scores else 0.0
        verified = max_overlap >= self.min_entity_overlap

        result = {
            "verified": verified,
            "overlap_scores": overlap_scores,
            "max_overlap": max_overlap,
        }

        if not verified:
            revised = self._generate_revised_question(
                question, f"Keyword overlap {max_overlap:.2f} below threshold"
            )
            result["revised_question"] = revised

        return result

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from a question.

        Parameters
        ----------
        question : str
            The question text.

        Returns
        -------
        List[str]
            List of keywords.
        """
        keywords = []

        # Extract capitalized words (likely entities)
        words = question.split()
        for word in words:
            # Remove punctuation
            cleaned = word.strip(".,?!:;\"'")
            # Keep if starts with capital or is long enough
            if cleaned and (cleaned[0].isupper() or len(cleaned) > 5):
                keywords.append(cleaned)

        return keywords

    def _generate_revised_question(self, question: str, reason: str) -> str:
        """Generate a revised question using LLM.

        Parameters
        ----------
        question : str
            Original question.
        reason : str
            Reason for revision.

        Returns
        -------
        str
            Revised question.
        """
        # Build prompt for LLM
        observation = {
            "task": (
                f"Rewrite or expand the following question to improve evidence retrieval.\n"
                f"Original question: {question}\n"
                f"Reason for revision: {reason}\n"
                f"Provide only the revised question, without explanation."
            )
        }

        # Generate revised question
        revised = super().act(observation)

        # Clean up response
        revised = revised.strip()

        # Remove common prefixes if present
        prefixes = ["Revised question:", "New question:", "Question:"]
        for prefix in prefixes:
            if revised.startswith(prefix):
                revised = revised[len(prefix) :].strip()

        return revised
