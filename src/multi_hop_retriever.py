#!/usr/bin/env python3
"""
Multi-hop Retrieval Enhancement Module

This module implements sophisticated multi-hop retrieval strategies to improve
evidence collection for complex reasoning questions, particularly those requiring
multiple steps of information gathering and entity linking.

Key Features:
1. Progressive evidence collection with iterative refinement
2. Entity relationship mapping and follow-up query generation
3. Context-aware query reformulation based on partial evidence
4. Intelligent stopping criteria to prevent infinite loops
5. Evidence coherence validation across hops
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HopResult:
    """Represents the result of a single retrieval hop"""
    hop_number: int
    query: str
    evidence: List[str]
    entities_found: Set[str]
    confidence_score: float
    source_hop: Optional[int] = None  # Which hop generated this query

@dataclass
class MultiHopConfig:
    """Configuration for multi-hop retrieval"""
    max_hops: int = 3
    evidence_per_hop: int = 5
    min_confidence_threshold: float = 0.3
    entity_overlap_threshold: float = 0.2
    enable_backward_validation: bool = True
    max_follow_up_queries: int = 3

class EntityExtractor:
    """Enhanced entity extraction for multi-hop scenarios"""

    def __init__(self):
        # Enhanced entity patterns for multi-hop scenarios
        self.entity_patterns = [
            r'\b[A-Z][a-zA-Z\s]{2,30}(?=\s(?:is|was|born|died|founded|established|located|directed|starred|performed|played|managed|coached))',
            r'\b(?:Mr\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-zA-Z\s]{2,25}',
            r'\b[A-Z][a-zA-Z\s]{2,25}(?=\s(?:University|College|Institute|Academy|School))',
            r'\b[A-Z][a-zA-Z\s]{2,25}(?=\s(?:Tower|Building|Arena|Stadium|Theater|Theatre|Museum|Gallery))',
            r'\b[A-Z][a-zA-Z\s]{2,25}(?=\s(?:Company|Corporation|Entertainment|Productions|Pictures|Studios))',
            r'\b[A-Z][a-zA-Z\s&]{2,30}(?=\s(?:band|group|team|organization|society))',
        ]

        # Relationship indicators for follow-up queries
        self.relationship_indicators = {
            'temporal': ['when', 'year', 'date', 'time', 'during', 'before', 'after', 'until'],
            'spatial': ['where', 'located', 'based', 'in', 'at', 'from', 'city', 'country'],
            'causal': ['why', 'because', 'due to', 'caused by', 'resulted in'],
            'descriptive': ['what', 'which', 'how', 'describe', 'characteristics'],
            'comparative': ['vs', 'versus', 'compared to', 'different', 'same', 'similar'],
            'quantitative': ['how many', 'how much', 'number', 'amount', 'size', 'capacity']
        }

    def extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract entities from text using enhanced patterns"""
        entities = set()

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update([match.strip() for match in matches if len(match.strip()) >= 3])

        # Additional named entity patterns
        # Names in quotes
        quoted_entities = re.findall(r'"([A-Z][a-zA-Z\s]{2,30})"', text)
        entities.update(quoted_entities)

        # Capitalized sequences (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in proper_nouns:
            if len(noun) >= 4 and noun not in ['This', 'That', 'These', 'Those', 'What', 'Who', 'Where', 'When', 'Why', 'How']:
                entities.add(noun)

        return entities

    def detect_query_relationships(self, query: str) -> List[str]:
        """Detect relationship types in the query for follow-up generation"""
        relationships = []
        query_lower = query.lower()

        for rel_type, indicators in self.relationship_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                relationships.append(rel_type)

        return relationships

class MultiHopRetriever:
    """Advanced multi-hop retrieval system"""

    def __init__(self,
                 retrieval_manager,
                 query_expander=None,
                 config: MultiHopConfig = None):
        self.rm = retrieval_manager
        self.query_expander = query_expander
        self.config = config or MultiHopConfig()
        self.entity_extractor = EntityExtractor()

        # Tracking for analysis
        self.hop_history = []
        self.entity_graph = defaultdict(set)

    def retrieve_multi_hop(self,
                          initial_query: str,
                          context: List[str] = None) -> Dict[str, Any]:
        """
        Perform multi-hop retrieval starting from initial query

        Returns:
            Dict containing all evidence, hop details, and metadata
        """
        logger.info(f"🔄 Starting multi-hop retrieval for: {initial_query}")

        # Initialize tracking
        self.hop_history = []
        self.entity_graph = defaultdict(set)
        all_evidence = []
        seen_evidence = set()

        # Hop 1: Initial query and expanded versions
        hop1_result = self._perform_initial_hop(initial_query)
        all_evidence.extend(hop1_result.evidence)
        seen_evidence.update([ev[:100].lower() for ev in hop1_result.evidence])

        # Track entities found in first hop
        initial_entities = self.entity_extractor.extract_entities_from_text(initial_query)
        hop1_entities = self._extract_entities_from_evidence(hop1_result.evidence)

        # Update entity graph
        for entity in hop1_entities:
            self.entity_graph[entity].update(initial_entities)

        current_entities = hop1_entities
        previous_queries = {initial_query}

        # Perform additional hops
        for hop_num in range(2, self.config.max_hops + 1):
            logger.info(f"🔄 Performing hop {hop_num}")

            # Generate follow-up queries based on current evidence
            follow_up_queries = self._generate_follow_up_queries(
                initial_query, current_entities, hop1_result.evidence[-3:], hop_num
            )

            if not follow_up_queries:
                logger.info(f"No follow-up queries generated for hop {hop_num}, stopping")
                break

            hop_evidence = []
            hop_entities = set()

            # Execute follow-up queries
            for query in follow_up_queries[:self.config.max_follow_up_queries]:
                if query in previous_queries:
                    continue

                previous_queries.add(query)

                # Retrieve evidence for this follow-up query
                query_evidence = self.rm.retrieve(query, top_k=self.config.evidence_per_hop)

                # Filter out duplicate evidence
                new_evidence = []
                for ev in query_evidence:
                    ev_key = ev[:100].lower()
                    if ev_key not in seen_evidence:
                        seen_evidence.add(ev_key)
                        new_evidence.append(ev)

                hop_evidence.extend(new_evidence)

                # Extract entities from new evidence
                query_entities = self._extract_entities_from_evidence(new_evidence)
                hop_entities.update(query_entities)

            if not hop_evidence:
                logger.info(f"No new evidence found in hop {hop_num}, stopping")
                break

            # Create hop result
            hop_result = HopResult(
                hop_number=hop_num,
                query="; ".join(follow_up_queries[:3]),
                evidence=hop_evidence,
                entities_found=hop_entities,
                confidence_score=self._calculate_hop_confidence(hop_evidence, current_entities)
            )

            self.hop_history.append(hop_result)
            all_evidence.extend(hop_evidence)

            # Update entity graph and current entities
            for entity in hop_entities:
                self.entity_graph[entity].update(current_entities)
            current_entities = hop_entities

            # Check stopping criteria
            if hop_result.confidence_score < self.config.min_confidence_threshold:
                logger.info(f"Confidence score {hop_result.confidence_score:.3f} below threshold, stopping")
                break

        # Perform backward validation if enabled
        if self.config.enable_backward_validation:
            all_evidence = self._validate_evidence_coherence(all_evidence, initial_query)

        # Compile final results
        result = {
            'evidence': all_evidence[:15],  # Limit total evidence
            'hop_count': len(self.hop_history) + 1,
            'hop_details': self.hop_history,
            'entity_graph': dict(self.entity_graph),
            'total_entities_found': len(set().union(*[hop.entities_found for hop in self.hop_history])),
            'confidence_scores': [hop.confidence_score for hop in self.hop_history]
        }

        logger.info(f"✅ Multi-hop retrieval completed: {result['hop_count']} hops, {len(result['evidence'])} total evidence")
        return result

    def _perform_initial_hop(self, query: str) -> HopResult:
        """Perform the initial retrieval hop"""
        if self.query_expander:
            # Use query expansion for the initial hop
            expanded = self.query_expander.expand_query(query)

            all_evidence = []
            seen = set()

            # Retrieve using expanded queries
            for exp_query in expanded.expanded[:3]:
                evidence = self.rm.retrieve(exp_query, top_k=self.config.evidence_per_hop)
                for ev in evidence:
                    if ev[:100].lower() not in seen:
                        seen.add(ev[:100].lower())
                        all_evidence.append(ev)
        else:
            # Simple retrieval without expansion
            all_evidence = self.rm.retrieve(query, top_k=self.config.evidence_per_hop)

        entities = self._extract_entities_from_evidence(all_evidence)

        hop_result = HopResult(
            hop_number=1,
            query=query,
            evidence=all_evidence,
            entities_found=entities,
            confidence_score=0.8  # High confidence for initial hop
        )

        self.hop_history.append(hop_result)
        return hop_result

    def _generate_follow_up_queries(self,
                                  original_query: str,
                                  entities: Set[str],
                                  recent_evidence: List[str],
                                  hop_number: int) -> List[str]:
        """Generate intelligent follow-up queries based on entities and evidence"""
        follow_ups = []

        # Extract relationships from original query
        relationships = self.entity_extractor.detect_query_relationships(original_query)

        # Generate entity-specific follow-up queries
        for entity in list(entities)[:5]:  # Limit to top 5 entities
            for rel_type in relationships[:2]:  # Top 2 relationship types
                if rel_type == 'temporal':
                    follow_ups.extend([
                        f"When was {entity} established founded created born?",
                        f"What year did {entity} start begin occur happen?",
                    ])
                elif rel_type == 'spatial':
                    follow_ups.extend([
                        f"Where is {entity} located based situated?",
                        f"What city country location is {entity} from in?",
                    ])
                elif rel_type == 'descriptive':
                    follow_ups.extend([
                        f"What is {entity} known for famous about?",
                        f"Who is {entity} associated with connected to?",
                    ])
                elif rel_type == 'comparative':
                    follow_ups.extend([
                        f"What are the characteristics features of {entity}?",
                        f"How is {entity} similar different compared to others?",
                    ])
                elif rel_type == 'quantitative':
                    follow_ups.extend([
                        f"What is the size capacity number amount of {entity}?",
                        f"How many people items does {entity} have contain?",
                    ])

        # Generate evidence-based follow-up queries
        if recent_evidence:
            # Extract key phrases from recent evidence
            evidence_text = " ".join(recent_evidence[-2:])  # Use last 2 pieces of evidence
            evidence_entities = self.entity_extractor.extract_entities_from_text(evidence_text)

            for evidence_entity in list(evidence_entities)[:3]:
                if evidence_entity not in entities:  # New entity discovered
                    follow_ups.append(f"Tell me about {evidence_entity} details information facts")

        # Remove duplicates and limit results
        unique_follow_ups = list(dict.fromkeys(follow_ups))  # Preserve order while removing duplicates
        return unique_follow_ups[:self.config.max_follow_up_queries * 2]  # Generate extra to filter later

    def _extract_entities_from_evidence(self, evidence_list: List[str]) -> Set[str]:
        """Extract entities from a list of evidence"""
        all_entities = set()

        for evidence in evidence_list:
            entities = self.entity_extractor.extract_entities_from_text(evidence)
            all_entities.update(entities)

        return all_entities

    def _calculate_hop_confidence(self, evidence: List[str], previous_entities: Set[str]) -> float:
        """Calculate confidence score for a hop based on entity overlap and evidence quality"""
        if not evidence or not previous_entities:
            return 0.0

        # Calculate entity overlap between hop and previous entities
        current_entities = self._extract_entities_from_evidence(evidence)

        if not current_entities:
            return 0.1

        overlap_count = len(current_entities.intersection(previous_entities))
        overlap_ratio = overlap_count / len(current_entities)

        # Factor in evidence length and content quality
        avg_length = sum(len(ev) for ev in evidence) / len(evidence)
        length_score = min(avg_length / 200, 1.0)  # Normalize to 0-1

        # Combine scores
        confidence = (overlap_ratio * 0.6) + (length_score * 0.4)
        return min(confidence, 1.0)

    def _validate_evidence_coherence(self, evidence_list: List[str], original_query: str) -> List[str]:
        """Validate evidence coherence and filter out irrelevant pieces"""
        if len(evidence_list) <= 5:
            return evidence_list  # Skip validation for small sets

        # Extract key terms from original query
        query_entities = self.entity_extractor.extract_entities_from_text(original_query)
        query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', original_query.lower()))

        scored_evidence = []

        for evidence in evidence_list:
            evidence_entities = self.entity_extractor.extract_entities_from_text(evidence)
            evidence_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', evidence.lower()))

            # Calculate relevance score
            entity_overlap = len(query_entities.intersection(evidence_entities))
            term_overlap = len(query_terms.intersection(evidence_terms))

            relevance_score = (entity_overlap * 2) + term_overlap
            scored_evidence.append((relevance_score, evidence))

        # Sort by relevance and return top pieces
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        return [evidence for score, evidence in scored_evidence[:12]]  # Return top 12

    def get_hop_summary(self) -> Dict[str, Any]:
        """Get a summary of the multi-hop retrieval process"""
        return {
            'total_hops': len(self.hop_history),
            'avg_confidence': sum(hop.confidence_score for hop in self.hop_history) / len(self.hop_history) if self.hop_history else 0,
            'total_entities': len(set().union(*[hop.entities_found for hop in self.hop_history])) if self.hop_history else 0,
            'hop_queries': [hop.query for hop in self.hop_history],
            'entity_progression': [len(hop.entities_found) for hop in self.hop_history]
        }

if __name__ == "__main__":
    # Simple test
    print("Multi-hop Retriever Module loaded successfully!")

    # Test entity extraction
    extractor = EntityExtractor()
    test_query = "Were Scott Derrickson and Ed Wood of the same nationality?"
    entities = extractor.extract_entities_from_text(test_query)
    print(f"Entities in '{test_query}': {entities}")

    relationships = extractor.detect_query_relationships(test_query)
    print(f"Relationships detected: {relationships}")