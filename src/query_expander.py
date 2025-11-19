#!/usr/bin/env python3
"""Query expansion module for improving evidence retrieval.

This module implements various query expansion strategies to improve
the quality and coverage of evidence retrieval, particularly for
entity-based and multi-hop questions.
"""

from __future__ import annotations

import re
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """Represents an expanded query with metadata."""

    original: str
    expanded: List[str]
    entities: List[str]
    query_type: str
    keywords: List[str]


class QueryExpander:
    """Expands queries to improve retrieval coverage and quality.

    The QueryExpander uses multiple strategies:
    1. Entity identification and context expansion
    2. Question type-specific keyword augmentation
    3. Synonym and related term generation
    4. Multi-hop query decomposition
    """

    def __init__(self):
        """Initialize the query expander."""
        # Common question patterns
        self.who_patterns = [
            r'\bwho\b',
            r'\bwhom\b',
            r'\bwhose\b',
            r'person',
            r'people',
            r'director',
            r'actor',
            r'author',
            r'founder',
            r'creator',
            r'artist',
            r'musician',
            r'politician',
            r'scientist'
        ]

        self.what_patterns = [
            r'\bwhat\b',
            r'name of',
            r'title',
            r'called',
            r'known as',
            r'position',
            r'role',
            r'job',
            r'occupation'
        ]

        self.where_patterns = [
            r'\bwhere\b',
            r'location',
            r'place',
            r'city',
            r'country',
            r'region',
            r'area',
            r'neighborhood',
            r'based in',
            r'located'
        ]

        self.when_patterns = [
            r'\bwhen\b',
            r'date',
            r'year',
            r'time',
            r'period',
            r'age',
            r'born',
            r'founded',
            r'established',
            r'created'
        ]

        # Domain-specific keyword mappings
        self.domain_keywords = {
            'government': ['ambassador', 'diplomat', 'minister', 'secretary', 'chief', 'protocol',
                          'administration', 'cabinet', 'department', 'office', 'official'],
            'entertainment': ['actor', 'actress', 'director', 'producer', 'film', 'movie',
                            'television', 'show', 'series', 'star', 'role', 'portrayed'],
            'music': ['musician', 'singer', 'band', 'album', 'song', 'artist',
                     'composer', 'performer', 'group', 'record'],
            'sports': ['athlete', 'player', 'team', 'league', 'championship', 'game',
                      'arena', 'stadium', 'coach', 'competition'],
            'business': ['company', 'corporation', 'CEO', 'founder', 'business',
                        'organization', 'enterprise', 'firm', 'executive'],
            'education': ['university', 'college', 'school', 'professor', 'student',
                         'academic', 'education', 'institution', 'campus'],
            'geography': ['city', 'country', 'region', 'area', 'location', 'place',
                         'district', 'neighborhood', 'town', 'village']
        }

    def expand_query(self, query: str, context: List[str] = None) -> ExpandedQuery:
        """Expand a query using multiple strategies.

        Parameters
        ----------
        query : str
            The original query to expand.
        context : List[str], optional
            Previous evidence or context that can inform expansion.

        Returns
        -------
        ExpandedQuery
            The expanded query with metadata.
        """
        # Extract entities and keywords
        entities = self.extract_entities(query)
        keywords = self.extract_keywords(query)
        query_type = self.classify_query_type(query)

        # Generate expanded queries
        expanded = []

        # 1. Original query
        expanded.append(query)

        # 2. Entity-based expansions
        for entity in entities:
            expanded.extend(self._expand_entity_query(entity, query_type))

        # 3. Keyword-based expansions
        expanded.extend(self._expand_keyword_query(query, keywords, query_type))

        # 4. Domain-specific expansions
        domain = self._detect_domain(query)
        if domain:
            expanded.extend(self._expand_domain_query(query, domain, entities))

        # 5. Context-informed expansions (if context provided)
        if context:
            expanded.extend(self._expand_with_context(query, context, entities))

        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for q in expanded:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_expanded.append(q)

        return ExpandedQuery(
            original=query,
            expanded=unique_expanded[:10],  # Limit to top 10 expansions
            entities=entities,
            query_type=query_type,
            keywords=keywords
        )

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using pattern matching.

        Parameters
        ----------
        text : str
            Text to extract entities from.

        Returns
        -------
        List[str]
            List of extracted entities.
        """
        entities = []

        # Common question words and stop words to exclude
        question_words = {
            'who', 'what', 'where', 'when', 'which', 'why', 'how', 'whose',
            'are', 'is', 'was', 'were', 'did', 'do', 'does', 'can', 'could',
            'would', 'should', 'will', 'the', 'a', 'an', 'in', 'on', 'at',
            'to', 'for', 'of', 'by', 'from', 'with'
        }

        # Pattern 1: Capitalized sequences (potential names)
        # Match 2-4 capitalized words (names, places, organizations)
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
        matches = re.findall(capitalized_pattern, text)

        for match in matches:
            # Filter out question words and common stop words
            if match.lower() not in question_words and len(match) > 2:
                entities.append(match)

        # Pattern 2: Quoted strings (often titles or names)
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)
        entities.extend(quoted_matches)

        # Pattern 3: Words after "called", "named", "known as"
        named_pattern = r'(?:called|named|known as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        named_matches = re.findall(named_pattern, text)
        entities.extend(named_matches)

        # Pattern 4: Proper nouns with multiple capitals (e.g., "Big Stone Gap", "Local H")
        # This handles cases like band names, movie titles
        multiword_pattern = r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b'
        multiword_matches = re.findall(multiword_pattern, text)
        for match in multiword_matches:
            if match.lower() not in [e.lower() for e in entities]:
                entities.append(match)

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen and entity_lower not in question_words:
                seen.add(entity_lower)
                unique_entities.append(entity)

        return unique_entities

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text.

        Parameters
        ----------
        text : str
            Text to extract keywords from.

        Returns
        -------
        List[str]
            List of extracted keywords.
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
            'for', 'with', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'that', 'this', 'these', 'those'
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))

    def classify_query_type(self, query: str) -> str:
        """Classify the query type based on patterns.

        Parameters
        ----------
        query : str
            The query to classify.

        Returns
        -------
        str
            The query type (who, what, where, when, comparison, multi_hop, other).
        """
        query_lower = query.lower()

        # Check for comparison questions
        if any(word in query_lower for word in ['same', 'different', 'compare', 'versus', 'vs', 'both', 'either', 'older', 'younger', 'larger', 'smaller']):
            return 'comparison'

        # Check for specific question types
        if any(re.search(pattern, query_lower) for pattern in self.who_patterns):
            return 'who'
        elif any(re.search(pattern, query_lower) for pattern in self.where_patterns):
            return 'where'
        elif any(re.search(pattern, query_lower) for pattern in self.when_patterns):
            return 'when'
        elif any(re.search(pattern, query_lower) for pattern in self.what_patterns):
            return 'what'

        # Check for multi-hop (contains multiple question components)
        question_markers = len(re.findall(r'\?', query))
        if question_markers > 1 or len(query.split(' and ')) > 1:
            return 'multi_hop'

        return 'other'

    def _expand_entity_query(self, entity: str, query_type: str) -> List[str]:
        """Generate entity-based query expansions.

        Parameters
        ----------
        entity : str
            The entity to expand queries for.
        query_type : str
            The type of query.

        Returns
        -------
        List[str]
            List of expanded queries.
        """
        expansions = []

        # Basic entity queries
        expansions.append(f"{entity} biography")
        expansions.append(f"{entity} information")

        # Type-specific expansions
        if query_type == 'who':
            expansions.extend([
                f"{entity} career",
                f"{entity} background",
                f"{entity} profile",
                f"who is {entity}"
            ])
        elif query_type == 'where':
            expansions.extend([
                f"{entity} location",
                f"{entity} based in",
                f"{entity} address",
                f"where is {entity} located"
            ])
        elif query_type == 'when':
            expansions.extend([
                f"{entity} founded",
                f"{entity} established",
                f"{entity} created",
                f"{entity} born",
                f"{entity} date"
            ])
        elif query_type == 'what':
            expansions.extend([
                f"{entity} description",
                f"{entity} details",
                f"{entity} facts"
            ])

        return expansions

    def _expand_keyword_query(self, query: str, keywords: List[str], query_type: str) -> List[str]:
        """Generate keyword-based query expansions.

        Parameters
        ----------
        query : str
            The original query.
        keywords : List[str]
            Extracted keywords.
        query_type : str
            The query type.

        Returns
        -------
        List[str]
            List of expanded queries.
        """
        expansions = []

        # Combine top keywords
        if len(keywords) >= 2:
            expansions.append(" ".join(keywords[:3]))
            expansions.append(" ".join(keywords[:2]))

        return expansions

    def _detect_domain(self, query: str) -> str | None:
        """Detect the domain of a query.

        Parameters
        ----------
        query : str
            The query to analyze.

        Returns
        -------
        str | None
            The detected domain, or None if no clear domain.
        """
        query_lower = query.lower()

        # Count domain-specific keywords
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score

        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return None

    def _expand_domain_query(self, query: str, domain: str, entities: List[str]) -> List[str]:
        """Generate domain-specific query expansions.

        Parameters
        ----------
        query : str
            The original query.
        domain : str
            The detected domain.
        entities : List[str]
            Extracted entities.

        Returns
        -------
        List[str]
            List of domain-specific expanded queries.
        """
        expansions = []

        # Get domain keywords
        domain_keywords = self.domain_keywords.get(domain, [])

        # Add domain context to entity queries
        for entity in entities[:2]:  # Limit to top 2 entities
            for keyword in domain_keywords[:3]:  # Top 3 domain keywords
                expansions.append(f"{entity} {keyword}")

        return expansions

    def _expand_with_context(self, query: str, context: List[str], entities: List[str]) -> List[str]:
        """Generate context-informed query expansions.

        Parameters
        ----------
        query : str
            The original query.
        context : List[str]
            Previous evidence or context.
        entities : List[str]
            Extracted entities from query.

        Returns
        -------
        List[str]
            List of context-informed expanded queries.
        """
        expansions = []

        # Extract entities from context
        context_entities = set()
        for ctx in context[:3]:  # Use recent context
            context_entities.update(self.extract_entities(ctx))

        # Combine query entities with context entities
        for q_entity in entities[:2]:
            for c_entity in list(context_entities)[:2]:
                if q_entity.lower() != c_entity.lower():
                    expansions.append(f"{q_entity} {c_entity}")

        return expansions


def test_query_expander():
    """Test the query expander with sample queries."""
    expander = QueryExpander()

    test_queries = [
        "What government position was held by Shirley Temple?",
        "Who directed the romantic comedy Big Stone Gap?",
        "Where is the Laleli Mosque located?",
        "When was the band Local H formed?",
        "Are Scott Derrickson and Ed Wood of the same nationality?"
    ]

    print("Query Expansion Test Results:")
    print("=" * 60)

    for query in test_queries:
        result = expander.expand_query(query)
        print(f"\nOriginal: {result.original}")
        print(f"Type: {result.query_type}")
        print(f"Entities: {result.entities}")
        print(f"Keywords: {result.keywords[:5]}")
        print(f"Expanded queries ({len(result.expanded)}):")
        for i, exp_query in enumerate(result.expanded[:5], 1):
            print(f"  {i}. {exp_query}")


if __name__ == "__main__":
    test_query_expander()