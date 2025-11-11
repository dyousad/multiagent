"""Retriever agent for evidence retrieval using RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent import Agent
# Try to use cached retrieval manager first, fall back to regular one
try:
    from cached_retrieval_manager import CachedRetrievalManager as RetrievalManager
    USING_CACHE = True
except ImportError:
    from retrieval_manager import RetrievalManager
    USING_CACHE = False


class RetrieverAgent(Agent):
    """Agent specialized in retrieving evidence passages for questions.

    This agent:
    - Uses RAG (Retrieval-Augmented Generation) to find relevant passages
    - Retrieves evidence from a corpus using semantic search
    - Reranks results by keyword overlap for better relevance
    - Returns top-k evidence passages
    """

    def __init__(
        self,
        agent_id: str = "retriever",
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        rerank: bool = True,
    ):
        """Initialize the retriever agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        retriever_config : Optional[Dict[str, Any]]
            Configuration for the RetrievalManager.
        top_k : int
            Number of evidence passages to return after reranking.
        rerank : bool
            Whether to rerank results by keyword overlap.
        """
        super().__init__(agent_id=agent_id, role="retriever")

        # Initialize retrieval manager
        retriever_config = retriever_config or {}
        # Retrieve more initially if we're going to rerank
        initial_top_k = retriever_config.get("top_k", 10 if rerank else top_k)
        retriever_config["top_k"] = initial_top_k

        if USING_CACHE:
            print(f"✓ Using CachedRetrievalManager (fast loading)")

        self.rm = RetrievalManager(**retriever_config)
        self.top_k = top_k
        self.rerank = rerank

    def act(self, observation: Dict[str, Any]) -> str:
        """Retrieve evidence for the given question.

        Parameters
        ----------
        observation : Dict[str, Any]
            Must contain 'sub_question' or 'question' or 'task' key.

        Returns
        -------
        str
            Formatted evidence passages.
        """
        # Extract question from observation
        question = (
            observation.get("sub_question")
            or observation.get("question")
            or observation.get("task", "")
        )

        if not question:
            return "Error: No question provided in observation"

        # Retrieve evidence
        evidence_list = self.rm.retrieve(question, top_k=self.rm.top_k)

        # Rerank by keyword overlap if enabled
        if self.rerank:
            evidence_list = self._rerank_by_keywords(question, evidence_list)

        # Take top-k after reranking
        final_evidence = evidence_list[: self.top_k]

        # Record contribution
        self.record_contribution(f"Retrieved {len(final_evidence)} evidence passages")

        # Format output
        formatted = self._format_evidence(final_evidence)
        return formatted

    def retrieve_evidence(self, question: str) -> Dict[str, Any]:
        """Retrieve evidence and return structured result.

        Parameters
        ----------
        question : str
            The question to retrieve evidence for.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'evidence': List of evidence passages
            - 'path': List of evidence previews (first 80 chars)
            - 'count': Number of evidence passages
        """
        # Retrieve evidence
        evidence_list = self.rm.retrieve(question, top_k=self.rm.top_k)

        # Rerank if enabled
        if self.rerank:
            evidence_list = self._rerank_by_keywords(question, evidence_list)

        # Take top-k
        final_evidence = evidence_list[: self.top_k]

        return {
            "evidence": final_evidence,
            "path": [e[:80] + "..." if len(e) > 80 else e for e in final_evidence],
            "count": len(final_evidence),
        }

    def _rerank_by_keywords(
        self, question: str, evidence_list: List[str]
    ) -> List[str]:
        """Rerank evidence by keyword overlap with the question.

        Parameters
        ----------
        question : str
            The question.
        evidence_list : List[str]
            List of evidence passages to rerank.

        Returns
        -------
        List[str]
            Reranked evidence list.
        """
        # Extract keywords (alphabetic words from question)
        keywords = [
            word.lower() for word in question.split() if word.isalpha() and len(word) > 2
        ]

        # Score each evidence by keyword overlap
        scored = []
        for evidence in evidence_list:
            evidence_lower = evidence.lower()
            score = sum(1 for keyword in keywords if keyword in evidence_lower)
            scored.append((score, evidence))

        # Sort by score (descending)
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return sorted evidence
        return [evidence for score, evidence in scored]

    def _format_evidence(self, evidence_list: List[str]) -> str:
        """Format evidence passages for output.

        Parameters
        ----------
        evidence_list : List[str]
            List of evidence passages.

        Returns
        -------
        str
            Formatted evidence string.
        """
        if not evidence_list:
            return "No evidence found."

        formatted_parts = [f"Retrieved {len(evidence_list)} evidence passages:\n"]

        for i, evidence in enumerate(evidence_list, 1):
            # Truncate very long passages
            if len(evidence) > 300:
                evidence = evidence[:297] + "..."
            formatted_parts.append(f"{i}. {evidence}\n")

        return "\n".join(formatted_parts)
