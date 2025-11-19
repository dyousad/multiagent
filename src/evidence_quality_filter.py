#!/usr/bin/env python3
"""
Evidence Quality Filter Module

This module implements sophisticated evidence quality assessment and filtering
to improve the relevance and coherence of retrieved information. It addresses
issues where retrieval systems return large volumes of evidence but with
inconsistent quality and relevance.

Key Features:
1. Content quality assessment (informativeness, completeness)
2. Query relevance scoring with multiple metrics
3. Evidence coherence evaluation across retrieved passages
4. Temporal consistency checking for time-sensitive queries
5. Entity consistency validation
6. Contradiction detection between evidence pieces
7. Source diversity and reliability assessment
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvidenceQualityScore:
    """Comprehensive quality score for a piece of evidence"""
    content_quality: float      # 0-1: Informativeness, completeness
    query_relevance: float      # 0-1: How well it matches the query
    entity_consistency: float   # 0-1: Entity coherence with other evidence
    temporal_consistency: float # 0-1: Time-related consistency
    contradiction_penalty: float # 0-1: Penalty for contradictions
    source_reliability: float   # 0-1: Source quality assessment

    overall_score: float = 0.0

    def __post_init__(self):
        """Calculate overall score as weighted average"""
        weights = {
            'content_quality': 0.25,
            'query_relevance': 0.30,
            'entity_consistency': 0.15,
            'temporal_consistency': 0.10,
            'contradiction_penalty': 0.10,
            'source_reliability': 0.10
        }

        self.overall_score = (
            self.content_quality * weights['content_quality'] +
            self.query_relevance * weights['query_relevance'] +
            self.entity_consistency * weights['entity_consistency'] +
            self.temporal_consistency * weights['temporal_consistency'] +
            (1 - self.contradiction_penalty) * weights['contradiction_penalty'] +
            self.source_reliability * weights['source_reliability']
        )

@dataclass
class FilterConfig:
    """Configuration for evidence quality filtering"""
    min_content_score: float = 0.3
    min_relevance_score: float = 0.4
    max_contradiction_penalty: float = 0.7
    enable_entity_consistency: bool = True
    enable_temporal_filtering: bool = True
    enable_contradiction_detection: bool = True
    diversity_threshold: float = 0.8  # For removing near-duplicates
    min_evidence_length: int = 20
    max_evidence_length: int = 2000

class EvidenceQualityFilter:
    """Advanced evidence quality assessment and filtering system"""

    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()

        # Patterns for quality assessment
        self.quality_indicators = {
            'positive': [
                r'\b(born|died|founded|established|located|created|built|opened|closed)\s+(?:in|on|during)\s+\d{4}',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|%|percent)',
                r'\b(?:directed|produced|starred|written|composed|designed|invented)\s+by\b',
                r'\b(?:first|second|third|last|final|initial|original|primary)\b',
                r'\b(?:according to|based on|reported by|confirmed by)\b',
                r'\b(?:university|college|company|corporation|organization|institution)\b',
            ],
            'negative': [
                r'\b(?:maybe|perhaps|possibly|allegedly|reportedly|supposedly)\b',
                r'\b(?:unknown|unclear|uncertain|unconfirmed|unverified)\b',
                r'\b(?:rumor|speculation|theory|hypothesis|assumption)\b',
                r'\[citation needed\]',
                r'\bstub\b.*article',
                r'\b(?:see also|external links|references)\b',
            ]
        }

        # Entity extraction patterns
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][A-Za-z\s&]{3,30}(?:University|College|Company|Corporation|Inc|Ltd)\b',
            r'\b(?:Mr\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        ]

        # Temporal indicators
        self.temporal_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',
            r'\b(?:in|during|from|until|since|before|after)\s+\d{4}',
        ]

    def filter_evidence(self,
                       evidence_list: List[str],
                       query: str,
                       context: List[str] = None) -> Dict[str, Any]:
        """
        Filter evidence based on quality metrics

        Parameters:
        -----------
        evidence_list : List[str]
            List of evidence passages to filter
        query : str
            Original query for relevance assessment
        context : List[str], optional
            Additional context for consistency checking

        Returns:
        --------
        Dict[str, Any]
            Filtered evidence with quality scores and metadata
        """

        if not evidence_list:
            return {
                'filtered_evidence': [],
                'quality_scores': [],
                'filter_stats': {'removed': 0, 'kept': 0, 'filter_reasons': {}}
            }

        logger.info(f"🔍 Filtering {len(evidence_list)} evidence pieces for query: {query[:60]}...")

        # Score each evidence piece
        scored_evidence = []
        filter_stats = {'removed': 0, 'kept': 0, 'filter_reasons': defaultdict(int)}

        for i, evidence in enumerate(evidence_list):
            # Calculate quality score
            quality_score = self._assess_evidence_quality(evidence, query, evidence_list, context)

            # Apply filters
            keep_evidence, filter_reason = self._should_keep_evidence(evidence, quality_score)

            if keep_evidence:
                scored_evidence.append((evidence, quality_score))
                filter_stats['kept'] += 1
            else:
                filter_stats['removed'] += 1
                filter_stats['filter_reasons'][filter_reason] += 1

        # Sort by overall quality score
        scored_evidence.sort(key=lambda x: x[1].overall_score, reverse=True)

        # Apply diversity filtering to remove near-duplicates
        final_evidence = self._apply_diversity_filter(scored_evidence)

        filtered_evidence = [ev for ev, score in final_evidence]
        quality_scores = [score for ev, score in final_evidence]

        logger.info(f"✅ Filtered to {len(filtered_evidence)} high-quality evidence pieces")

        return {
            'filtered_evidence': filtered_evidence,
            'quality_scores': quality_scores,
            'filter_stats': filter_stats,
            'avg_quality_score': sum(score.overall_score for score in quality_scores) / len(quality_scores) if quality_scores else 0,
            'quality_distribution': self._get_quality_distribution(quality_scores)
        }

    def _assess_evidence_quality(self,
                                evidence: str,
                                query: str,
                                all_evidence: List[str],
                                context: List[str] = None) -> EvidenceQualityScore:
        """Assess the quality of a single evidence piece"""

        # 1. Content Quality Assessment
        content_quality = self._assess_content_quality(evidence)

        # 2. Query Relevance Assessment
        query_relevance = self._assess_query_relevance(evidence, query)

        # 3. Entity Consistency Assessment
        entity_consistency = self._assess_entity_consistency(evidence, all_evidence) if self.config.enable_entity_consistency else 1.0

        # 4. Temporal Consistency Assessment
        temporal_consistency = self._assess_temporal_consistency(evidence, query) if self.config.enable_temporal_filtering else 1.0

        # 5. Contradiction Detection
        contradiction_penalty = self._detect_contradictions(evidence, all_evidence) if self.config.enable_contradiction_detection else 0.0

        # 6. Source Reliability Assessment
        source_reliability = self._assess_source_reliability(evidence)

        return EvidenceQualityScore(
            content_quality=content_quality,
            query_relevance=query_relevance,
            entity_consistency=entity_consistency,
            temporal_consistency=temporal_consistency,
            contradiction_penalty=contradiction_penalty,
            source_reliability=source_reliability
        )

    def _assess_content_quality(self, evidence: str) -> float:
        """Assess the informational quality of evidence content"""
        score = 0.5  # Base score

        # Length-based quality (sweet spot around 100-500 chars)
        length = len(evidence)
        if length < self.config.min_evidence_length:
            score -= 0.3
        elif length > self.config.max_evidence_length:
            score -= 0.2
        elif 100 <= length <= 500:
            score += 0.2

        # Positive quality indicators
        for pattern in self.quality_indicators['positive']:
            if re.search(pattern, evidence, re.IGNORECASE):
                score += 0.1

        # Negative quality indicators
        for pattern in self.quality_indicators['negative']:
            if re.search(pattern, evidence, re.IGNORECASE):
                score -= 0.15

        # Information density (number of facts per unit length)
        fact_density = self._calculate_fact_density(evidence)
        score += min(fact_density * 0.1, 0.2)

        return max(0.0, min(1.0, score))

    def _assess_query_relevance(self, evidence: str, query: str) -> float:
        """Assess how relevant the evidence is to the query"""

        # Extract key terms from query and evidence
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        evidence_terms = set(re.findall(r'\b\w{3,}\b', evidence.lower()))

        if not query_terms:
            return 0.5

        # Term overlap score
        term_overlap = len(query_terms.intersection(evidence_terms)) / len(query_terms)

        # Entity overlap score
        query_entities = self._extract_entities(query)
        evidence_entities = self._extract_entities(evidence)

        if query_entities:
            entity_overlap = len(query_entities.intersection(evidence_entities)) / len(query_entities)
        else:
            entity_overlap = 0

        # Combine scores
        relevance_score = (term_overlap * 0.6) + (entity_overlap * 0.4)

        # Boost for exact phrase matches
        query_phrases = [phrase.strip() for phrase in query.split() if len(phrase.strip()) > 3]
        for phrase in query_phrases:
            if phrase.lower() in evidence.lower():
                relevance_score += 0.1

        return min(1.0, relevance_score)

    def _assess_entity_consistency(self, evidence: str, all_evidence: List[str]) -> float:
        """Assess entity consistency across evidence pieces"""

        evidence_entities = self._extract_entities(evidence)

        if not evidence_entities:
            return 1.0  # No entities to be inconsistent about

        # Check entity consistency across other evidence
        consistent_entities = 0
        total_checks = 0

        for other_evidence in all_evidence[:10]:  # Check against up to 10 other pieces
            if other_evidence == evidence:
                continue

            other_entities = self._extract_entities(other_evidence)

            for entity in evidence_entities:
                if entity in other_entities:
                    consistent_entities += 1
                total_checks += 1

        if total_checks == 0:
            return 1.0

        return consistent_entities / total_checks

    def _assess_temporal_consistency(self, evidence: str, query: str) -> float:
        """Assess temporal consistency in evidence"""

        # Extract years from evidence and query
        evidence_years = set(re.findall(r'\b(19|20)\d{2}\b', evidence))
        query_years = set(re.findall(r'\b(19|20)\d{2}\b', query))

        if not evidence_years and not query_years:
            return 1.0  # No temporal information to check

        if not evidence_years:
            return 0.8  # Evidence lacks temporal information when query has it

        if not query_years:
            return 1.0  # Query doesn't specify time, so any time is fine

        # Check if evidence years are reasonable given query years
        query_year_ints = [int(year) for year in query_years]
        evidence_year_ints = [int(year) for year in evidence_years]

        min_query_year = min(query_year_ints) if query_year_ints else 1900
        max_query_year = max(query_year_ints) if query_year_ints else 2030

        # Allow reasonable temporal range (±20 years for context)
        reasonable_evidence = [
            year for year in evidence_year_ints
            if min_query_year - 20 <= year <= max_query_year + 20
        ]

        if len(reasonable_evidence) == len(evidence_year_ints):
            return 1.0
        elif reasonable_evidence:
            return len(reasonable_evidence) / len(evidence_year_ints)
        else:
            return 0.3  # Temporally inconsistent

    def _detect_contradictions(self, evidence: str, all_evidence: List[str]) -> float:
        """Detect contradictions between evidence pieces"""

        # Simple contradiction patterns
        contradiction_patterns = [
            (r'\b(?:born|established|founded)\s+(?:in|on)\s+(\d{4})', 'birth_year'),
            (r'\b(?:died|ended|closed)\s+(?:in|on)\s+(\d{4})', 'death_year'),
            (r'\b(?:is|was)\s+(?:a|an)\s+([^.]{1,50})(?:\.|,)', 'description'),
            (r'\b(?:located|based|situated)\s+in\s+([^.]{1,30})(?:\.|,)', 'location'),
        ]

        evidence_facts = {}
        for pattern, fact_type in contradiction_patterns:
            matches = re.findall(pattern, evidence, re.IGNORECASE)
            if matches:
                evidence_facts[fact_type] = matches[0].strip().lower()

        if not evidence_facts:
            return 0.0  # No extractable facts to contradict

        # Check against other evidence
        contradictions = 0
        total_comparisons = 0

        for other_evidence in all_evidence[:5]:  # Check against up to 5 other pieces
            if other_evidence == evidence:
                continue

            other_facts = {}
            for pattern, fact_type in contradiction_patterns:
                matches = re.findall(pattern, other_evidence, re.IGNORECASE)
                if matches:
                    other_facts[fact_type] = matches[0].strip().lower()

            # Compare facts
            for fact_type, value in evidence_facts.items():
                if fact_type in other_facts:
                    total_comparisons += 1
                    if value != other_facts[fact_type] and not self._are_compatible_values(value, other_facts[fact_type], fact_type):
                        contradictions += 1

        if total_comparisons == 0:
            return 0.0

        return contradictions / total_comparisons

    def _assess_source_reliability(self, evidence: str) -> float:
        """Assess the apparent reliability of the evidence source"""
        score = 0.7  # Base score

        # Positive reliability indicators
        reliable_indicators = [
            r'\b(?:university|academic|research|institute|official|government)\b',
            r'\b(?:published|peer-reviewed|journal|encyclopedia|database)\b',
            r'\b(?:according to|based on|reported by)\b.*(?:study|research|analysis)',
        ]

        for pattern in reliable_indicators:
            if re.search(pattern, evidence, re.IGNORECASE):
                score += 0.1

        # Negative reliability indicators
        unreliable_indicators = [
            r'\b(?:blog|opinion|editorial|rumor|gossip|social media)\b',
            r'\b(?:unverified|unconfirmed|allegedly|supposedly|reportedly)\b',
            r'\[citation needed\]',
            r'\b(?:fan site|wiki|forum|comment)\b',
        ]

        for pattern in unreliable_indicators:
            if re.search(pattern, evidence, re.IGNORECASE):
                score -= 0.15

        return max(0.1, min(1.0, score))

    def _should_keep_evidence(self, evidence: str, score: EvidenceQualityScore) -> Tuple[bool, str]:
        """Determine whether to keep evidence based on quality thresholds"""

        # Check minimum thresholds
        if score.content_quality < self.config.min_content_score:
            return False, "low_content_quality"

        if score.query_relevance < self.config.min_relevance_score:
            return False, "low_relevance"

        if score.contradiction_penalty > self.config.max_contradiction_penalty:
            return False, "high_contradictions"

        if len(evidence) < self.config.min_evidence_length:
            return False, "too_short"

        if len(evidence) > self.config.max_evidence_length:
            return False, "too_long"

        return True, "kept"

    def _apply_diversity_filter(self, scored_evidence: List[Tuple[str, EvidenceQualityScore]]) -> List[Tuple[str, EvidenceQualityScore]]:
        """Remove near-duplicate evidence to ensure diversity"""

        if len(scored_evidence) <= 3:
            return scored_evidence

        final_evidence = []

        for evidence, score in scored_evidence:
            # Check similarity against already selected evidence
            is_diverse = True

            for selected_evidence, _ in final_evidence:
                similarity = self._calculate_text_similarity(evidence, selected_evidence)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                final_evidence.append((evidence, score))

        return final_evidence

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text pieces"""
        # Simple word overlap-based similarity
        words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        return overlap / total if total > 0 else 0.0

    def _calculate_fact_density(self, text: str) -> float:
        """Calculate the density of factual information in text"""

        fact_indicators = [
            r'\b\d+\b',  # Numbers
            r'\b(?:born|died|founded|established|located|created)\b',  # Factual verbs
            r'\b(?:in|on|during|from|to|until|since)\s+\d{4}\b',  # Time expressions
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
        ]

        fact_count = 0
        for pattern in fact_indicators:
            fact_count += len(re.findall(pattern, text))

        # Normalize by text length (facts per 100 characters)
        return (fact_count / max(len(text), 1)) * 100

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text"""
        entities = set()

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            entities.update([match.strip() for match in matches if len(match.strip()) >= 3])

        return entities

    def _are_compatible_values(self, value1: str, value2: str, fact_type: str) -> bool:
        """Check if two values are compatible (not contradictory)"""

        # For years, allow small differences (OCR errors, etc.)
        if fact_type in ['birth_year', 'death_year']:
            try:
                year1 = int(re.search(r'\d{4}', value1).group())
                year2 = int(re.search(r'\d{4}', value2).group())
                return abs(year1 - year2) <= 1
            except:
                return False

        # For descriptions and locations, check for partial matches
        if fact_type in ['description', 'location']:
            return value1 in value2 or value2 in value1 or len(set(value1.split()) & set(value2.split())) >= 2

        return False

    def _get_quality_distribution(self, scores: List[EvidenceQualityScore]) -> Dict[str, float]:
        """Get quality score distribution statistics"""
        if not scores:
            return {}

        overall_scores = [score.overall_score for score in scores]

        return {
            'min_score': min(overall_scores),
            'max_score': max(overall_scores),
            'mean_score': sum(overall_scores) / len(overall_scores),
            'high_quality_count': sum(1 for score in overall_scores if score >= 0.7),
            'medium_quality_count': sum(1 for score in overall_scores if 0.4 <= score < 0.7),
            'low_quality_count': sum(1 for score in overall_scores if score < 0.4)
        }

    def get_filter_summary(self, filter_result: Dict[str, Any]) -> str:
        """Generate a human-readable summary of filtering results"""

        stats = filter_result['filter_stats']
        quality_dist = filter_result['quality_distribution']

        summary = f"""
📊 Evidence Quality Filter Summary:
  • Original evidence pieces: {stats['kept'] + stats['removed']}
  • Kept after filtering: {stats['kept']}
  • Removed: {stats['removed']}
  • Average quality score: {filter_result['avg_quality_score']:.3f}

🎯 Quality Distribution:
  • High quality (≥0.7): {quality_dist.get('high_quality_count', 0)}
  • Medium quality (0.4-0.7): {quality_dist.get('medium_quality_count', 0)}
  • Low quality (<0.4): {quality_dist.get('low_quality_count', 0)}

🔍 Filter Reasons:
"""
        for reason, count in stats['filter_reasons'].items():
            summary += f"  • {reason.replace('_', ' ').title()}: {count}\n"

        return summary

if __name__ == "__main__":
    # Simple test
    print("Evidence Quality Filter Module loaded successfully!")

    # Test with sample evidence
    filter_system = EvidenceQualityFilter()

    sample_evidence = [
        "The University of California was founded in 1868 in Berkeley, California. It is a public research university.",
        "Maybe the university was established sometime in the 1800s, but I'm not sure about the exact date.",
        "UC Berkeley is located in Berkeley, California and was founded on March 23, 1868. The campus covers 1,232 acres.",
        "See also: List of universities. External links: Official website.",
        "According to historical records, the University of California system began in 1868 with the Berkeley campus."
    ]

    test_query = "When was UC Berkeley founded?"

    result = filter_system.filter_evidence(sample_evidence, test_query)
    print(f"\nFiltered {len(sample_evidence)} to {len(result['filtered_evidence'])} evidence pieces")
    print(f"Average quality: {result['avg_quality_score']:.3f}")

    for i, evidence in enumerate(result['filtered_evidence']):
        score = result['quality_scores'][i]
        print(f"\nEvidence {i+1} (Score: {score.overall_score:.3f}):")
        print(f"  {evidence[:100]}...")
        print(f"  Content: {score.content_quality:.2f}, Relevance: {score.query_relevance:.2f}")