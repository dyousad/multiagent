"""Enhanced Retriever agent v3 with better ranking and more results."""

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


class RetrieverAgentV3(Agent):
    """Enhanced retriever agent with improved ranking.

    Key improvements in v3:
    - Higher top-k (10 instead of 5) for better recall
    - Multi-stage ranking: semantic → keyword → diversity
    - Better handling of entity-specific queries
    """

    def __init__(
        self,
        agent_id: str = "retriever_v3",
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 15,  # Increased from 10 to 15
        rerank: bool = True,
    ):
        """Initialize the enhanced retriever agent.

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
        # Retrieve more initially (2.5x) for better reranking
        initial_top_k = retriever_config.get("top_k", 30 if rerank else top_k)
        retriever_config["top_k"] = initial_top_k

        if USING_CACHE:
            print(f"✓ Using CachedRetrievalManager for {agent_id}")

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

        # Enhanced reranking if enabled
        if self.rerank:
            evidence_list = self._enhanced_rerank(question, evidence_list)

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

        # Enhanced rerank if enabled
        if self.rerank:
            evidence_list = self._enhanced_rerank(question, evidence_list)

        # Take top-k
        final_evidence = evidence_list[: self.top_k]

        return {
            "evidence": final_evidence,
            "path": [e[:80] + "..." if len(e) > 80 else e for e in final_evidence],
            "count": len(final_evidence),
        }

    def _enhanced_rerank(self, question: str, evidence_list: List[str]) -> List[str]:
        """Enhanced multi-stage reranking.

        Stage 1: Keyword overlap scoring
        Stage 2: Entity bonus (if question asks about specific entity)
        Stage 3: Diversity (avoid too similar passages)

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
        import re

        # Extract keywords (alphabetic words from question)
        keywords = [
            word.lower()
            for word in question.split()
            if word.isalpha() and len(word) > 2
        ]

        # Extract entities (capitalized words/phrases) from question
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, question)

        # Score each evidence
        scored = []
        for evidence in evidence_list:
            evidence_lower = evidence.lower()

            # Stage 1: Keyword overlap score
            keyword_score = sum(1 for keyword in keywords if keyword in evidence_lower)

            # Stage 2: Entity bonus (higher weight)
            entity_score = sum(
                3 for entity in entities if entity.lower() in evidence_lower
            )

            # Stage 3: Length bonus (prefer more substantial passages)
            length_score = min(len(evidence) / 200, 2.0)  # Cap at 2.0

            # Combined score
            total_score = keyword_score + entity_score + length_score

            scored.append((total_score, evidence))

        # Sort by score (descending)
        scored.sort(reverse=True, key=lambda x: x[0])

        # Apply diversity filter (remove very similar passages)
        diverse_evidence = self._apply_diversity_filter(
            [evidence for score, evidence in scored]
        )

        return diverse_evidence

    def _apply_diversity_filter(
        self, evidence_list: List[str], similarity_threshold: float = 0.7
    ) -> List[str]:
        """Apply diversity filter to avoid too similar passages.

        Parameters
        ----------
        evidence_list : List[str]
            Sorted list of evidence passages.
        similarity_threshold : float
            Threshold for considering passages similar (0-1).

        Returns
        -------
        List[str]
            Filtered list with diverse passages.
        """
        if len(evidence_list) <= 5:
            return evidence_list  # Keep all if small list

        diverse = []
        for evidence in evidence_list:
            # Check if this passage is too similar to already selected ones
            is_similar = False
            for selected in diverse:
                # Simple similarity: shared word ratio
                evidence_words = set(evidence.lower().split())
                selected_words = set(selected.lower().split())
                overlap = len(evidence_words & selected_words)
                total = len(evidence_words | selected_words)
                similarity = overlap / total if total > 0 else 0

                if similarity > similarity_threshold:
                    is_similar = True
                    break

            if not is_similar or len(diverse) < 3:
                # Always keep at least top 3, then apply diversity
                diverse.append(evidence)

        return diverse

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
            if len(evidence) > 400:
                evidence = evidence[:397] + "..."
            formatted_parts.append(f"{i}. {evidence}\n")

        return "\n".join(formatted_parts)
