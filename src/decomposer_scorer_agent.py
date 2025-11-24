"""Agent that synthesizes final answers from one or more decomposer outputs and computes Shapley scores for decomposers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import itertools
import random

from llm_agent import LLMAgent


class DecomposerScorerAgent(LLMAgent):
    """Uses an LLM to synthesize a final answer from decomposer outputs and evaluate coalitions.

    Methods
    -------
    synthesize_answer(decomposer_outputs, question)
        Return a string answer produced by the LLM given the selected decomposer outputs.
    compute_shapley(decomposer_outputs_map, question, ground_truth, use_mc=False, mc_samples=100)
        Compute Shapley values for each decomposer id. If use_mc True and number of decomposers
        is large, approximate via Monte Carlo sampling.
    """

    def __init__(self, agent_id: str = "decomposer_scorer", model_identifier: str = "deepseek-ai/DeepSeek-V3", max_tokens: int = 512, temperature: float = 0.0):
        system_prompt = (
            "You are an assistant that synthesizes final answers from multiple decompositions of a complex question.\n"
            "Given a list of decomposer outputs (sub-questions, keywords, and optional retrieved evidence), produce a concise final answer.\n"
            "When asked to evaluate subsets of decomposers, use only the information provided for that subset.\n"
            "Always respond with the final answer text only (no JSON) when synthesizing."
        )
        super().__init__(agent_id=agent_id, model_identifier=model_identifier, role="scorer", system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature)

    def synthesize_answer(self, decomposer_outputs: List[Dict[str, Any]], question: str) -> str:
        """Synthesize final answer using LLM from provided decomposer outputs.

        decomposer_outputs: list of dicts with keys 'sub_questions', 'keywords', 'pipeline_output' and 'final_answer' optional
        """
        parts = [f"Question: {question}\n"]
        parts.append("Available decomposer outputs:\n")
        for i, d in enumerate(decomposer_outputs):
            parts.append(f"--- Decomposer {i} ({d.get('decomposer','unknown')}) ---\n")
            subs = d.get('sub_questions') or []
            if subs:
                parts.append("Sub-questions:\n")
                for j, sq in enumerate(subs):
                    parts.append(f"{j+1}. {sq}\n")
            kws = d.get('keywords') or []
            if kws:
                parts.append("Keywords:\n")
                for k in kws:
                    parts.append(f"- {k}\n")
            # include any pipeline evidence fragment if available
            po = d.get('pipeline_output') or {}
            agent_outputs = po.get('agent_outputs') or {}
            if agent_outputs:
                parts.append("Evidence / agent outputs (short):\n")
                for k, v in list(agent_outputs.items())[:3]:
                    if isinstance(v, dict) and 'evidence' in v:
                        parts.append(f"- {k}: {(' '.join(v.get('evidence')[:2]))}\n")
                    else:
                        parts.append(f"- {k}: {str(v)[:200]}\n")
            # if decomposer already contains a final answer
            if d.get('final_answer'):
                parts.append(f"Decomposer final_answer: {d.get('final_answer')}\n")

        prompt = "\n".join(parts) + "\nBased on the available information, provide the best concise final answer to the question. If insufficient information, return an empty string."
        observation = {"task": prompt}
        ans = self.act(observation)
        return ans.strip()

    def _value_function(self, coalition_outputs: List[Dict], question: str, ground_truth: str) -> float:
        """Define the value of a coalition as the F1 score of the synthesized answer vs ground_truth."""
        synth = self.synthesize_answer(coalition_outputs, question)
        print("getting answer:",synth)
        # compute simple F1
        if not synth or not ground_truth:
            return 0.0
        pred_tokens = set(synth.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        if not pred_tokens or not truth_tokens:
            return 0.0
        common = pred_tokens & truth_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def compute_shapley(self, decomposer_outputs_map: Dict[str, Dict], question: str, ground_truth: str, use_mc: bool = False, mc_samples: int = 200) -> Dict[str, float]:
        """Compute Shapley values for each decomposer id.

        decomposer_outputs_map: mapping decomposer_id -> decomposer_output_dict
        If number of decomposers <= 6 (configurable), compute exact Shapley; otherwise approximate by Monte Carlo.
        """
        ids = list(decomposer_outputs_map.keys())
        n = len(ids)
        if n == 0:
            return {}

        # Precompute factorials
        fact = math.factorial
        shapley = {i: 0.0 for i in ids}

        # Helper to get coalition outputs by ids
        def outputs_for(subset_ids: List[str]) -> List[Dict]:
            return [decomposer_outputs_map[i] for i in subset_ids]

        # Exact computation if small n
        if not use_mc and n <= 6:
            for i in ids:
                phi = 0.0
                for r in range(0, n):
                    # sum over subsets of size r not containing i
                    from itertools import combinations
                    for S in combinations([x for x in ids if x != i], r):
                        S = list(S)
                        print("\ntesting combinition:", S)
                        weight = fact(r) * fact(n - r - 1) / fact(n)
                        v_S = self._value_function(outputs_for(S), question, ground_truth)
                        v_Si = self._value_function(outputs_for(S + [i]), question, ground_truth)
                        phi += weight * (v_Si - v_S)
                shapley[i] = phi
            return shapley

        # Monte Carlo approximation
        counts = {i: 0.0 for i in ids}
        totals = {i: 0.0 for i in ids}
        for _ in range(mc_samples):
            perm = ids[:]  # deterministic order then shuffle
            random.shuffle(perm)
            seen = []
            prev_val = 0.0
            for j, pid in enumerate(perm):
                # value of coalition seen U {pid}
                val = self._value_function(outputs_for(seen + [pid]), question, ground_truth)
                marg = val - prev_val
                totals[pid] += marg
                counts[pid] += 1
                prev_val = val
                seen.append(pid)
        for i in ids:
            shapley[i] = totals[i] / max(1, counts[i])
        return shapley
