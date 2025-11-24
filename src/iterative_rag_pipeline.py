"""Enhanced controller with iterative retrieval for multi-hop questions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from decomposer_agent import DecomposerAgent
from retriever_agent_v3 import RetrieverAgentV3
from evidence_verifier_agent import EvidenceVerifierAgent
from reasoner_agent_v3 import ReasonerAgentV3


class IterativeRAGPipeline:
    """RAG pipeline with iterative retrieval for multi-hop reasoning.

    Key features:
    - First sub-question is answered independently
    - Answer from first sub-question is used to enhance retrieval for second
    - Supports chaining information across multiple hops
    """

    def __init__(
        self,
        sub_questions,
        retriever: RetrieverAgentV3,
        verifier: EvidenceVerifierAgent,
        reasoner: ReasonerAgentV3,
        keywords : List[str] = [],
        use_iterative: bool = True
    ):
        """Initialize the iterative RAG pipeline.

        Parameters
        ----------
        decomposer : DecomposerAgent
            Agent for question decomposition.
        retriever : RetrieverAgentV3
            Agent for evidence retrieval.
        verifier : EvidenceVerifierAgent
            Agent for evidence verification.
        reasoner : ReasonerAgentV3
            Agent for answer synthesis.
        use_iterative : bool
            Whether to use iterative retrieval (default: True).
        """
        self.sub_questions = sub_questions
        self.retriever = retriever
        self.verifier = verifier
        self.reasoner = reasoner
        self.use_iterative = use_iterative
        self.keywords = keywords

    def run(self, question: str) -> Dict[str, Any]:
        """Run the RAG pipeline with iterative retrieval.

        Parameters
        ----------
        question : str
            The main question to answer.

        Returns
        -------
        Dict[str, Any]
            Pipeline output containing sub-questions, evidence, and answers.
        """
        # Step 1: Decompose question
        # decomposition = self.decomposer.decompose_question(question)
        # sub_questions = decomposition.get("sub_questions", [])
        sub_questions = self.sub_questions
        print("\n subquestion counts: ",len(sub_questions))
        
        if not self.sub_questions:
            return {
                "sub_questions": [],
                "agent_outputs": {},
                "final_answer": "unknown",
                "error": "Failed to decompose question"
            }

        # Step 2: Process sub-questions iteratively
        agent_outputs = {}
        intermediate_answers = {}  # Store answers for context

        for i, sub_q in enumerate(sub_questions):
            # Build enhanced query using previous answers
            enhanced_query = self._build_enhanced_query(
                sub_q,
                i,
                intermediate_answers,
                sub_questions[:i]  # Previous sub-questions for context
            )
            sub_q = self.sub_questions[i]
            # Retrieve evidence with enhanced query
            retrieval_result = self.retriever.retrieve_evidence(enhanced_query)
            evidence = retrieval_result.get("evidence", [])
            evidence_path = retrieval_result.get("path", [])

            # Verify evidence
            verification_result = self.verifier.verify_evidence(
                question=sub_q,
                evidence=evidence
            )
            verified = verification_result.get("verified", False)

            # Try to extract intermediate answer for next iteration
            if evidence and self.use_iterative and i < len(sub_questions):
                # Use reasoner to get intermediate answer
                temp_answer = self.reasoner.act({
                    "sub_results": {sub_q: {"evidence": evidence}},
                    "main_question": sub_q
                })
                print("\nAnswering subquestion: ",sub_q)
                print("Generating answer: ",temp_answer)
                intermediate_answers[i] = temp_answer

            # Store results
            agent_outputs[sub_q] = {
                "evidence": evidence,
                "evidence_path": evidence_path,
                "verified": verified,
                "overlap_scores": verification_result.get("overlap_scores", []),
                "max_overlap": verification_result.get("max_overlap", 0.0)
            }

        # Step 3: Synthesize final answer
        final_answer_result = self.reasoner.synthesize_answer(
            main_question=question,
            sub_results=agent_outputs
        )
        final_answer = final_answer_result.get("final_answer", "unknown")

        return {
            "sub_questions": sub_questions,
            "agent_outputs": agent_outputs,
            "final_answer": final_answer,
            "intermediate_answers": intermediate_answers,
            "reasoning": final_answer_result.get("reasoning", "")
        }

    def _build_enhanced_query(
        self,
        current_sub_q: str,
        current_index: int,
        intermediate_answers: Dict[int, str],
        previous_sub_qs: List[str]
    ) -> str:
        """Build enhanced query using previous answers.

        Parameters
        ----------
        current_sub_q : str
            Current sub-question.
        intermediate_answers : Dict[str, str]
            Previous sub-questions and their answers.
        previous_sub_qs : List[str]
            Previous sub-questions (in order).

        Returns
        -------
        str
            Enhanced query for retrieval.
        """
        if not self.use_iterative or not intermediate_answers:
            if current_index < len(self.keywords):
                return self.keywords[current_index]
            return current_sub_q

        # current_keywords = self.keywords[current_index]
        
        parsed_question = self.resolve_question(
            current_index,
            intermediate_answers
            )
        self.sub_questions[current_index] = parsed_question

        parsed_keywords = self.resolve_keyword(
            current_index,
            intermediate_answers)
        
        return parsed_keywords
    
        # Build context from previous answers
        context_parts = []
        for prev_q in previous_sub_qs:
            if prev_q in intermediate_answers:
                answer = intermediate_answers[prev_q]
                # Only add if answer is useful (not "unknown")
                if answer and answer != "unknown" and len(answer) > 0:
                    # Extract key entity/fact from answer
                    context_parts.append(answer)

        if context_parts:
            # Enhance current query with context
            # Format: "Given that [context], [current question]"
            context_str = ", ".join(context_parts[:2])  # Use at most 2 previous answers
            enhanced_query = f"Given {context_str}, {current_sub_q}"
            return enhanced_query
        else:
            return current_sub_q
        
        
    @staticmethod
    def _find_dependencies(question):
        """Find all ANSWER_X dependencies in a question"""
        pattern = r'ANSWER_(\w+)'
        return re.findall(pattern, question)
    
    def resolve_question(self, q_id,intermediate_answers: Dict[int, str], use_answers=True):
        """Resolve a question by replacing ANSWER_X placeholders"""
        if q_id > len(self.sub_questions):
            raise ValueError(f"Question {q_id} not found")
        
        question = self.sub_questions[q_id]
        
        if not use_answers:
            return question
        
        def replace_match(match):
            ref_id =int( match.group(1))-1
            ref_ans = intermediate_answers.get(ref_id)
            if ref_ans :
                return ref_ans
            else:
                raise ValueError(f"Question {q_id} not complete")
                return match.group(0)  # Keep original if answer not available
        
        # Replace all ANSWER_X references
        resolved = re.sub(r'ANSWER_(\w+)', replace_match, question)
        return resolved
    
    def resolve_keyword(self, q_id,intermediate_answers: Dict[int, str], use_answers=True):
        """Resolve a keyword by replacing ANSWER_X placeholders"""
        if q_id > len(self.sub_questions):
            raise ValueError(f"Question {q_id} not found")
        
        keyword = self.keywords[q_id]
        
        if not use_answers:
            return keyword
        
        def replace_match(match):
            ref_id = int( match.group(1))-1
            ref_ans = intermediate_answers.get(ref_id)
            
            if ref_ans :
                return ref_ans
            else:
                raise ValueError(f"Question {q_id} not complete")
                return match.group(0)  # Keep original if answer not available
        
        # Replace all ANSWER_X references
        resolved = re.sub(r'ANSWER_(\w+)', replace_match, keyword)
        return resolved