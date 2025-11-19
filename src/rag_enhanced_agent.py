"""RAG-Enhanced Agent that combines LLM capabilities with retrieval."""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

from api_models import APILanguageModel, get_api_model
from agent import Agent
from query_expander import QueryExpander
from multi_hop_retriever import MultiHopRetriever, MultiHopConfig
from evidence_quality_filter import EvidenceQualityFilter, FilterConfig

# Try to use cached retrieval manager first, fall back to regular one
try:
    from cached_retrieval_manager import CachedRetrievalManager as RetrievalManager
    USING_CACHE = True
except ImportError:
    from retrieval_manager import RetrievalManager
    USING_CACHE = False


class RAGEnhancedAgent(Agent):
    """An agent that combines LLM capabilities with RAG retrieval.

    This agent:
    - Uses an external LLM for reasoning and response generation
    - Has retrieval capabilities to gather relevant information
    - Combines retrieved evidence with LLM reasoning
    - Suitable for knowledge-intensive tasks
    """

    def __init__(
        self,
        agent_id: str,
        model_identifier: str = "deepseek-ai/DeepSeek-V3",
        role: str = "assistant",
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,  # Fewer passages per agent to reduce noise
        rerank: bool = True,
        use_query_expansion: bool = True,  # Enable query expansion by default
        use_multi_hop: bool = False,  # Enable multi-hop retrieval for complex questions
        multi_hop_config: Optional[MultiHopConfig] = None,
        use_quality_filter: bool = True,  # Enable evidence quality filtering
        quality_filter_config: Optional[FilterConfig] = None,
    ):
        """Initialize a RAG-enhanced agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        model_identifier : str
            Model identifier for API.
        role : str
            The role or persona of this agent.
        system_prompt : Optional[str]
            System prompt to guide the agent's behavior.
        max_tokens : int
            Maximum tokens for LLM generation.
        temperature : float
            Sampling temperature for LLM.
        retriever_config : Optional[Dict[str, Any]]
            Configuration for the RetrievalManager.
        top_k : int
            Number of evidence passages to retrieve.
        rerank : bool
            Whether to rerank results by keyword overlap.
        use_query_expansion : bool
            Whether to use query expansion for better retrieval.
        use_multi_hop : bool
            Whether to use multi-hop retrieval for complex reasoning.
        multi_hop_config : Optional[MultiHopConfig]
            Configuration for multi-hop retrieval behavior.
        use_quality_filter : bool
            Whether to use evidence quality filtering for better relevance.
        quality_filter_config : Optional[FilterConfig]
            Configuration for evidence quality filtering behavior.
        """
        super().__init__(agent_id=agent_id, role=role)
        self.model_identifier = model_identifier
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.rerank = rerank
        self.use_query_expansion = use_query_expansion
        self.use_multi_hop = use_multi_hop
        self.use_quality_filter = use_quality_filter

        # Initialize query expander if enabled
        if self.use_query_expansion:
            self.query_expander = QueryExpander()
        else:
            self.query_expander = None

        # Initialize multi-hop retriever if enabled
        if self.use_multi_hop:
            self.multi_hop_config = multi_hop_config or MultiHopConfig()
            self.multi_hop_retriever = None  # Will be initialized after retrieval manager
        else:
            self.multi_hop_retriever = None

        # Initialize quality filter if enabled
        if self.use_quality_filter:
            self.quality_filter_config = quality_filter_config or FilterConfig()
            self.quality_filter = EvidenceQualityFilter(self.quality_filter_config)
        else:
            self.quality_filter = None

        # Initialize LLM
        self.llm = get_api_model(model_identifier)

        # Get model name and provider safely
        model_name = getattr(self.llm, 'name', model_identifier)
        provider_name = getattr(self.llm, 'provider', 'Unknown')
        print(f"Loading API model '{model_identifier}' with name '{model_name}' and provider '{provider_name}'")

        # Set system prompt
        self.system_prompt = system_prompt or f"You are a {role}. Use the retrieved evidence to provide accurate, well-reasoned responses."

        # Initialize retrieval manager
        retriever_config = retriever_config or {}
        initial_top_k = retriever_config.get("top_k", 10 if rerank else top_k)
        retriever_config["top_k"] = initial_top_k

        if USING_CACHE:
            print(f"✓ {agent_id}: Using CachedRetrievalManager (fast loading)")

        try:
            self.rm = RetrievalManager(**retriever_config)
            self.has_retrieval = True

            # Initialize multi-hop retriever after retrieval manager is ready
            if self.use_multi_hop:
                self.multi_hop_retriever = MultiHopRetriever(
                    retrieval_manager=self.rm,
                    query_expander=self.query_expander,  # Reuse query expander if available
                    config=self.multi_hop_config
                )
                print(f"✓ {agent_id}: Multi-hop retrieval enabled")

        except Exception as e:
            print(f"⚠ {agent_id}: Retrieval manager initialization failed: {e}")
            self.has_retrieval = False

        # Store conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def retrieve_evidence(self, query: str) -> Dict[str, Any]:
        """Retrieve evidence passages for a given query using query expansion.

        Parameters
        ----------
        query : str
            The query to search for.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing retrieved evidence and metadata.
        """
        if not self.has_retrieval:
            return {"evidence": [], "evidence_path": [], "query": query}

        try:
            # Use multi-hop retrieval if enabled (for complex questions)
            if self.use_multi_hop and self.multi_hop_retriever:
                # Multi-hop retrieval performs both expansion and multi-step reasoning
                multi_hop_result = self.multi_hop_retriever.retrieve_multi_hop(query)

                retrieved_evidence = multi_hop_result['evidence']

                # Apply quality filtering to multi-hop results if enabled
                if self.use_quality_filter and self.quality_filter and retrieved_evidence:
                    filter_result = self.quality_filter.filter_evidence(retrieved_evidence, query)
                    final_evidence = filter_result['filtered_evidence'][:self.top_k]

                    # Quality metadata for multi-hop
                    quality_metadata = {
                        "quality_filter_used": True,
                        "original_evidence_count": len(retrieved_evidence),
                        "filtered_evidence_count": len(filter_result['filtered_evidence']),
                        "evidence_filtered_out": filter_result['filter_stats']['removed'],
                        "avg_quality_score": filter_result['avg_quality_score']
                    }
                else:
                    final_evidence = retrieved_evidence[:self.top_k]
                    quality_metadata = {"quality_filter_used": False}

                evidence_path = [ev[:80] + "..." if len(ev) > 80 else ev for ev in final_evidence]

                result = {
                    "evidence": final_evidence,
                    "evidence_path": evidence_path,
                    "query": query,
                    "num_retrieved": len(final_evidence),
                    "multi_hop_used": True,
                    "hop_count": multi_hop_result['hop_count'],
                    "entities_found": multi_hop_result['total_entities_found'],
                    "confidence_scores": multi_hop_result['confidence_scores']
                }

                # Add quality metadata
                result.update(quality_metadata)
                return result

            # Use query expansion if enabled (but not multi-hop)
            elif self.use_query_expansion and self.query_expander:
                all_evidence = []
                evidence_sources = []

                expanded = self.query_expander.expand_query(query)

                # Retrieve using multiple expanded queries
                for exp_query in expanded.expanded[:5]:  # Use top 5 expanded queries
                    evidence_list = self.rm.retrieve(exp_query, top_k=3)  # 3 per query
                    all_evidence.extend(evidence_list)
                    evidence_sources.append(f"Expanded: {exp_query[:40]}")

                # Deduplicate while preserving order
                seen = set()
                deduped_evidence = []
                for ev in all_evidence:
                    ev_lower = ev.lower()[:100]  # Use first 100 chars for dedup
                    if ev_lower not in seen:
                        seen.add(ev_lower)
                        deduped_evidence.append(ev)

                all_evidence = deduped_evidence

            else:
                # Original query only (no expansion, no multi-hop)
                all_evidence = self.rm.retrieve(query, top_k=10)
                evidence_sources = [f"Original: {query[:40]}"]

            if self.rerank and all_evidence:
                # Rerank by keyword overlap
                all_evidence = self._rerank_by_keywords(query, all_evidence)

            # Apply quality filtering if enabled
            if self.use_quality_filter and self.quality_filter and all_evidence:
                filter_result = self.quality_filter.filter_evidence(all_evidence, query)
                filtered_evidence = filter_result['filtered_evidence']

                # Store filter statistics for analysis
                filter_stats = filter_result['filter_stats']
                avg_quality = filter_result['avg_quality_score']

                # Use filtered evidence
                final_evidence = filtered_evidence[:self.top_k] if filtered_evidence else all_evidence[:self.top_k]

                # Additional metadata for quality filtering
                quality_metadata = {
                    "quality_filter_used": True,
                    "original_evidence_count": len(all_evidence),
                    "filtered_evidence_count": len(filtered_evidence),
                    "evidence_filtered_out": filter_stats['removed'],
                    "avg_quality_score": avg_quality
                }
            else:
                # No quality filtering
                final_evidence = all_evidence[:self.top_k] if all_evidence else []
                quality_metadata = {"quality_filter_used": False}

            evidence_path = [ev[:80] + "..." if len(ev) > 80 else ev for ev in final_evidence]

            result = {
                "evidence": final_evidence,
                "evidence_path": evidence_path,
                "query": query,
                "num_retrieved": len(final_evidence),
                "query_expansion_used": self.use_query_expansion,
                "num_expanded_queries": len(evidence_sources) if self.use_query_expansion else 1
            }

            # Add quality metadata to result
            result.update(quality_metadata)

            return result

        except Exception as e:
            print(f"⚠ {self.agent_id}: Retrieval failed for query '{query}': {e}")
            import traceback
            traceback.print_exc()
            return {"evidence": [], "evidence_path": [], "query": query}

    def _rerank_by_keywords(self, query: str, evidence_list: List[str]) -> List[str]:
        """Rerank results by keyword overlap with query."""
        try:
            import re
            query_keywords = set(re.findall(r'\w+', query.lower()))

            def keyword_score(text):
                text_keywords = set(re.findall(r'\w+', text.lower()))
                overlap = len(query_keywords & text_keywords)
                return overlap / len(query_keywords) if query_keywords else 0

            return sorted(evidence_list, key=keyword_score, reverse=True)
        except:
            return evidence_list

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Convenience method for retrieving evidence (backward compatibility).

        Parameters
        ----------
        query : str
            The query to search for.
        top_k : int, optional
            Number of results to return. If None, uses self.top_k.

        Returns
        -------
        List[str]
            List of evidence strings.
        """
        old_top_k = self.top_k
        if top_k is not None:
            self.top_k = top_k

        result = self.retrieve_evidence(query)

        self.top_k = old_top_k
        return result.get("evidence", [])

    def act(self, observation: str) -> str:
        """Generate a response based on observation with RAG enhancement.

        Parameters
        ----------
        observation : str or Dict
            The observation or question to respond to.

        Returns
        -------
        str
            The agent's response.
        """
        try:
            # Convert observation to string if it's a dict (for compatibility)
            if isinstance(observation, dict):
                query_text = observation.get('task', str(observation))
            else:
                query_text = str(observation)

            # Step 1: Retrieve relevant evidence
            evidence_result = self.retrieve_evidence(query_text)
            evidence = evidence_result.get("evidence", [])

            # Step 2: Construct context with evidence
            if evidence:
                evidence_context = "\n".join([
                    f"Evidence {i+1}: {ev}"
                    for i, ev in enumerate(evidence[:self.top_k])
                ])
                context = f"Question: {query_text}\n\nRetrieved Evidence:\n{evidence_context}\n\nPlease provide a well-reasoned answer based on the evidence above."
            else:
                context = f"Question: {query_text}\n\nNote: No specific evidence was retrieved. Please provide your best answer based on your knowledge."

            # Step 3: Build prompt similar to LLMAgent
            prompt_lines = [self.system_prompt, ""]

            if isinstance(observation, dict):
                # Handle dict observation format
                if "task" in observation:
                    prompt_lines.append(f"Task: {observation['task']}")
                    prompt_lines.append("")

                if "previous_responses" in observation:
                    prompt_lines.append("Previous agent responses:")
                    for resp in observation["previous_responses"]:
                        prompt_lines.append(f"  - {resp}")
                    prompt_lines.append("")

            prompt_lines.append(context)
            prompt = "\n".join(prompt_lines)

            # Step 4: Generate response using LLM (using generate method like LLMAgent)
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Step 5: Store conversation history
            self.conversation_history.append({
                "observation": query_text,
                "evidence_count": len(evidence),
                "response": response
            })

            return response

        except Exception as e:
            print(f"Error in {self.agent_id} act(): {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def get_response_with_evidence(self, observation: str) -> Dict[str, Any]:
        """Get response along with evidence metadata.

        Parameters
        ----------
        observation : str
            The observation or question to respond to.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing response, evidence, and metadata.
        """
        evidence_result = self.retrieve_evidence(observation)
        response = self.act(observation)

        return {
            "response": response,
            "evidence": evidence_result.get("evidence", []),
            "evidence_path": evidence_result.get("evidence_path", []),
            "num_retrieved": evidence_result.get("num_retrieved", 0),
            "agent_id": self.agent_id,
            "role": self.role
        }

    def reset(self):
        """Reset the agent's conversation history."""
        self.conversation_history = []