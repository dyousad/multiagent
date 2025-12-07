"""Experiment runner for HotpotQA with dynamic credit allocation."""

from __future__ import annotations

import json
import pprint
import sys
from pathlib import Path
from typing import Dict, List, Set, Union
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment_hotpotqa import HotpotQAEnvironment
from llm_agent import LLMAgent
from decomposer_agent import DecomposerAgent
from retriever_agent import RetrieverAgent
from evidence_verifier_agent import EvidenceVerifierAgent
from controller import MultiAgentController
from reward_manager import RewardManager
from environment import TaskResult

# try:
import spacy
SPACY_AVAILABLE = True
    # try:
nlp = spacy.load("en_core_web_sm")
#     except OSError:
#         print("Warning: spaCy model 'en_core_web_sm' not found.")
#         print("Please run: python -m spacy download en_core_web_sm")
#         SPACY_AVAILABLE = False
# except ImportError:
#     SPACY_AVAILABLE = False
#     print("Warning: spaCy not installed. Install with: pip install spacy)



def create_agents_with_rag(
    num_agents: int,
    model_identifier: str,
    agent_roles: List[str] = None,
    use_rag: bool = True,
    corpus_path: str = None,
    use_v3: bool = False,
    agent_model_identifiers: Union[List[str], Dict[str, str]] = None
) -> tuple[List[LLMAgent], DecomposerAgent]:
    """Create agents including decomposer, retriever, and verifier.

    Parameters
    ----------
    num_agents : int
        Number of main agents to create.
    model_identifier : str
        Default model identifier for LLM API.
    agent_roles : List[str]
        List of roles for agents.
    use_rag : bool
        Whether to include RAG agents (retriever and verifier).
    corpus_path : str
        Path to corpus for retrieval (optional).
    use_v3 : bool
        Whether to use v3 enhanced agents (Retriever v3, Reasoner v3).
    agent_model_identifiers : List[str] or Dict[str,str]
        Optional per-agent model identifiers. Can be a list aligned with
        `num_agents` (e.g. ["m1","m2","m3"]) or a dict mapping either
        agent ids (`"agent_0"`) or roles (`"planner"`, `"reasoner"`, `"verifier"`)
        to model identifiers. Dict form is recommended for specifying
        models for special agents like `verifier`, `reasoner`, `decomposer`.

    Returns
    -------
    tuple[List[LLMAgent], DecomposerAgent]
        Main agents and decomposer agent.
    """
    if agent_roles is None:
        agent_roles = []
        # default_roles = ["reasoner"]
        # agent_roles = (default_roles * ((num_agents // len(default_roles)) + 1))[:num_agents]

    # Create main agents
    agents = []
    for i in range(num_agents):
        role = agent_roles[i] if i < len(agent_roles) else "assistant"
        system_prompt = f"You are a {role}. Your goal is to help answer multi-hop questions by reasoning step by step."

        # Determine model identifier for this agent
        if isinstance(agent_model_identifiers, dict):
            # Prefer agent id, then role, then default
            model_id = agent_model_identifiers.get(f"agent_{i}",
                        agent_model_identifiers.get(role, model_identifier))
        elif isinstance(agent_model_identifiers, list):
            model_id = agent_model_identifiers[i] if i < len(agent_model_identifiers) else model_identifier
        else:
            model_id = model_identifier

        agent = LLMAgent(
            agent_id=f"agent_{i}",
            model_identifier=model_id,
            role=role,
            system_prompt=system_prompt,
            max_tokens=512,
            temperature=0.5
        )
        agents.append(agent)

    # Add RAG agents if requested
    if use_rag:
        try:
            # Create retriever agent (v3 or regular)
            retriever_config = {}
            if corpus_path:
                retriever_config["corpus_path"] = corpus_path
            else:
                # Try full corpus first (cached), fall back to small corpus
                full_corpus = "data/hotpotqa_corpus_full.json"
                small_corpus = "data/hotpotqa_corpus.json"
                from pathlib import Path
                if Path(full_corpus).exists():
                    retriever_config["corpus_path"] = full_corpus
                    print(f"✓ Using full corpus (cached): {full_corpus}")
                elif Path(small_corpus).exists():
                    retriever_config["corpus_path"] = small_corpus
                    print(f"⚠ Using small corpus: {small_corpus}")
                    print(f"  For better results, run: python scripts/build_full_corpus_cache.py")
                else:
                    print(f"Warning: Corpus file not found")
                    print("Please run: python scripts/build_full_corpus_cache.py")
                    print("Skipping RAG agents...")
                    raise FileNotFoundError(f"Corpus not found")

            if use_v3:
                from retriever_agent_v3 import RetrieverAgentV3
                retriever = RetrieverAgentV3(
                    agent_id="retriever",
                    retriever_config=retriever_config,
                    top_k=15,  # v3 uses more results (increased from 10)
                    rerank=True
                )
                print("✓ Added RetrieverAgentV3 (enhanced, top_k=15)")
            else:
                retriever = RetrieverAgent(
                    agent_id="retriever",
                    retriever_config=retriever_config,
                    top_k=5,
                    rerank=True
                )
                print("✓ Added RetrieverAgent")
            agents.append(retriever)

            # Determine verifier model if provided
            verifier_model = model_identifier
            if isinstance(agent_model_identifiers, dict):
                verifier_model = agent_model_identifiers.get("verifier", verifier_model)

            # Create evidence verifier agent
            verifier = EvidenceVerifierAgent(
                agent_id="verifier",
                model_identifier=verifier_model,
                max_tokens=256,
                temperature=0.7,
                min_entity_overlap=0.5,  # Added parameter
                use_spacy=True  # Enable spaCy if available
            )
            agents.append(verifier)
            print("✓ Added EvidenceVerifierAgent")

            # Create reasoner agent for answer synthesis (v3 or regular)
            reasoner_model = model_identifier
            if isinstance(agent_model_identifiers, dict):
                reasoner_model = agent_model_identifiers.get("reasoner", reasoner_model)

            if use_v3:
                from reasoner_agent_v3 import ReasonerAgentV3
                reasoner = ReasonerAgentV3(
                    agent_id="reasoner",
                    model_identifier=reasoner_model,
                    max_tokens=1024,
                    temperature=0.3  # Lower temperature for more consistent answers
                )
                print("✓ Added ReasonerAgentV3 (enhanced)")
            else:
                from reasoner_agent import ReasonerAgent
                reasoner = ReasonerAgent(
                    agent_id="reasoner",
                    model_identifier=reasoner_model,
                    max_tokens=2048,
                    temperature=0.6
                )
                print("✓ Added ReasonerAgent")
            agents.append(reasoner)

        except Exception as e:
            print(f"⚠ Could not create RAG agents: {e}")
            print("Continuing without RAG components...")

    # Create decomposer agent
    decomposer_model = model_identifier
    if isinstance(agent_model_identifiers, dict):
        decomposer_model = agent_model_identifiers.get("decomposer", decomposer_model)

    decomposer = DecomposerAgent(
        agent_id="decomposer",
        model_identifier=decomposer_model,
        max_tokens=512,
        temperature=0.4
    )

    return agents, decomposer


def run_hotpotqa_experiment(
    data_path: str = "data/hotpot_dev_fullwiki_v1.json",
    max_samples: int = 10,
    num_agents: int = 3,
    model_identifier: str = "deepseek-ai/DeepSeek-V3",
    use_decomposer: bool = True,
    use_rag: bool = True,
    use_dynamic_credit: bool = True,
    use_v3: bool = True,
    use_iterative: bool = True,  # New parameter for iterative RAG
    agent_model_identifiers: Union[List[str], Dict[str, str]] = None,
    output_dir: Path = Path("results/hotpotqa")
) -> Dict:
    """Run HotpotQA experiment with RAG and dynamic credit allocation.

    Parameters
    ----------
    data_path : str
        Path to HotpotQA data file.
    max_samples : int
        Number of samples to evaluate.
    num_agents : int
        Number of agents.
    model_identifier : str
        Default model identifier for LLM API (used when no per-agent model supplied).
    use_decomposer : bool
        Whether to use question decomposition.
    use_rag : bool
        Whether to use RAG (retriever and verifier agents).
    use_dynamic_credit : bool
        Whether to use dynamic credit allocation.
    use_v3 : bool
        Whether to use v3 enhanced agents.
    use_iterative : bool
        Whether to use iterative RAG (answers from earlier hops enhance later retrieval).
    agent_model_identifiers : List[str] or Dict[str,str]
        Optional per-agent model identifiers (list or dict). See `create_agents_with_rag`.
    output_dir : Path
        Output directory for results.

    Returns
    -------
    Dict
        Experiment results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running HotpotQA Experiment with RAG")
    print(f"  Data: {data_path}")
    print(f"  Samples: {max_samples}")
    print(f"  Agents: {num_agents}")
    print(f"  Model (default): {model_identifier}")
    print(f"  Decomposer: {use_decomposer}")
    print(f"  RAG: {use_rag}")
    print(f"  Dynamic Credit: {use_dynamic_credit}")
    print(f"  V3 Enhanced: {use_v3}")
    print(f"  Iterative RAG: {use_iterative}")
    print(f"{'='*70}\n")

    # Load environment
    env = HotpotQAEnvironment(
        data_path=data_path,
        max_samples=max_samples,
        split="validation"
    )

    # Create agents with RAG components

    # 支持多个 decomposer agent
    decomposer_agents = []
    if isinstance(agent_model_identifiers, dict):
        # 检查是否有多个 decomposer 模型
        decomposer_models = []
        for k, v in agent_model_identifiers.items():
            if k.startswith("decomposer"):
                decomposer_models.append((k, v))
        if not decomposer_models:
            decomposer_models = [("decomposer", agent_model_identifiers.get("decomposer", model_identifier))]
    else:
        decomposer_models = [("decomposer", model_identifier)]

    # 创建主 agents
    agents, _ = create_agents_with_rag(
        num_agents=num_agents,
        model_identifier=model_identifier,
        use_rag=use_rag,
        use_v3=use_v3,
        agent_model_identifiers=agent_model_identifiers
    )

    # 创建所有 decomposer agents
    from decomposer_agent import DecomposerAgent
    for name, model_id in decomposer_models:
        decomposer_agents.append(
            DecomposerAgent(
                agent_id=name,
                model_identifier=model_id,
                max_tokens=512,
                temperature=0.6
            )
        )

    # 为每个 decomposer 创建独立 reasoner agent
    reasoner_agents = []
    use_reasoner_v3 = use_v3
    for i, decomp in enumerate(decomposer_agents):
        reasoner_model_id = decomp.model_identifier
        if use_reasoner_v3:
            from reasoner_agent_v3 import ReasonerAgentV3
            reasoner = ReasonerAgentV3(
                agent_id=f"reasoner_{i}",
                model_identifier=reasoner_model_id,
                max_tokens=1024,
                temperature=0.3
            )
        else:
            from reasoner_agent import ReasonerAgent
            reasoner = ReasonerAgent(
                agent_id=f"reasoner_{i}",
                model_identifier=reasoner_model_id,
                max_tokens=2048,
                temperature=0.6
            )
        reasoner_agents.append(reasoner)
    print(f"✓ Added {len(reasoner_agents)} reasoner agents (one per decomposer)")

    # 创建一个评分 agent（用于基于 decomposer 输出合成答案并计算 Shapley）
    from decomposer_scorer_agent import DecomposerScorerAgent
    scorer_agent = DecomposerScorerAgent(
        agent_id="decomposer_scorer",
        model_identifier=agent_model_identifiers.get("scorer", model_identifier) if isinstance(agent_model_identifiers, dict) else model_identifier,
        max_tokens=512,
        temperature=0.0
    )
    print("✓ Added scorer")
    # Initialize reward manager
    reward_manager = RewardManager(base_reward=1.0, log_file = Path("results/logs/reward.log"))

    # Results tracking
    results = {
        "config": {
            "data_path": data_path,
            "max_samples": max_samples,
            "num_agents": num_agents,
            "model": model_identifier,
            "use_decomposer": use_decomposer,
            "use_rag": use_rag,
            "use_dynamic_credit": use_dynamic_credit,
        },
        "samples": [],
        "metrics": {
            "exact_match": [],
            "f1_scores": [],
            "credit_entropy_static": [],
            "credit_entropy_dynamic": [],
        }
    }

    # Evaluate each sample
    # print(env.get_num_samples())
    # tests = random.SystemRandom().sample(range(env.get_num_samples()),max_samples)
    

    for idx in range(env.get_num_samples()):
        print(f"\n--- Sample {idx + 1}/{max_samples} ---")

        env.set_current_task(idx)
        task_data = env.current_task_data

        print(f"Question: {task_data['question']}")
        print(f"Answer: {task_data['answer']}")
        
         # Find required agents
        retriever_agent = None
        verifier_agent = None
        reasoner_agent = None
        for agent in agents:
            if hasattr(agent, 'role'):
                if agent.role == 'retriever':
                    retriever_agent = agent
                elif agent.role in ['verifier', 'evidence_verifier']:
                    verifier_agent = agent
        

        # Helper to sum numeric usage fields into a single token count estimate
        def sum_usage(u: Dict[str, float]) -> float:
            if not u:
                return 0.0
            # prefer total_tokens if present
            try:
                return u["input_tokens"] + 6.0*u["output_tokens"]
            except Exception:
                return 0.0

        # Snapshot usage at sample start for shared agents
        scorer_start_usage = scorer_agent.llm.get_accumulated_usage() if getattr(scorer_agent, 'llm', None) else {}
        retriever_start_usage = retriever_agent.llm.get_accumulated_usage() if (retriever_agent and getattr(retriever_agent, 'llm', None)) else {}
        verifier_start_usage = verifier_agent.llm.get_accumulated_usage() if (verifier_agent and getattr(verifier_agent, 'llm', None)) else {}

       
        # 不再全局找 reasoner_agent，改为每个 decomposer 配对

        # 多 decomposer 分别调用 pipeline
        decomposer_results = []
        for i, decomp_agent in enumerate(decomposer_agents):
            print(f"\n>>> Using decomposer: {decomp_agent.model_identifier}")
            # snapshot usage for this decomposer+its reasoner before work
            before_tokens = 0.0
            try:
                before_tokens += sum_usage(decomp_agent.llm.get_accumulated_usage()) if getattr(decomp_agent, 'llm', None) else 0.0
            except Exception:
                pass
            reasoner_agent = reasoner_agents[i] if i < len(reasoner_agents) else None
            try:
                before_tokens += sum_usage(reasoner_agent.llm.get_accumulated_usage()) if (reasoner_agent and getattr(reasoner_agent, 'llm', None)) else 0.0
            except Exception:
                pass

            try:
                decomposition = decomp_agent.decompose_question(task_data['question'])
                sub_questions = decomposition.get("sub_questions", [])
                keywords = decomposition.get("keywords", [])
                evidence_paths = []
                pipeline_output = {}
                final_answer = ""

                # 每个 decomposer 用自己的 reasoner
                reasoner_agent = reasoner_agents[i] if i < len(reasoner_agents) else None

                if use_rag and use_iterative and use_v3 and all([retriever_agent, verifier_agent, reasoner_agent]):
                    from iterative_rag_pipeline import IterativeRAGPipeline
                    pipeline = IterativeRAGPipeline(
                        sub_questions=sub_questions,
                        retriever=retriever_agent,
                        verifier=verifier_agent,
                        reasoner=reasoner_agent,
                        keywords = keywords,
                        use_iterative=True
                    )
                    pipeline_output = pipeline.run(task_data['question'])
                    sub_questions = pipeline_output.get("sub_questions", [])
                    final_answer = pipeline_output.get("final_answer", "unknown")
                    agent_outputs_rag = pipeline_output.get("agent_outputs", {})
                    for sq, output in agent_outputs_rag.items():
                        if "evidence_path" in output:
                            evidence_paths.extend(output["evidence_path"])
                    print(f"Sub-questions: {len(sub_questions)}")
                    print(f"Evidence retrieved: {len(evidence_paths)} passages")
                    print(f"Iterative answers: {len(pipeline_output.get('intermediate_answers', {}))}")
                    print(f"Final answer: {final_answer[:100]}...")
                else:
                    # fallback to standard execution
                    print("⚠ Missing required agents for iterative pipeline, falling back to standard execution")
                    pipeline_output = {}
                    sub_questions = []
                    evidence_paths = []
                    final_answer = ""

                # compute tokens used by this decomposer (decomposer llm + its reasoner)
                after_tokens = 0.0
                try:
                    after_tokens += sum_usage(decomp_agent.llm.get_accumulated_usage()) if getattr(decomp_agent, 'llm', None) else 0.0
                except Exception:
                    pass
                try:
                    after_tokens += sum_usage(reasoner_agent.llm.get_accumulated_usage()) if (reasoner_agent and getattr(reasoner_agent, 'llm', None)) else 0.0
                except Exception:
                    pass
                tokens_used = max(0.0, after_tokens - before_tokens)

                # 结果收集
                decomposer_results.append( {
                    "sub_questions": sub_questions,
                    "evidence_paths": evidence_paths,
                    "pipeline_output": pipeline_output,
                    "final_answer": final_answer,
                    "decomposer" : decomp_agent.model_identifier,
                    "tokens_used": tokens_used,
                })
            except Exception as e:
                print(f"Warning: pipeline failed for decomposer {decomp_agent.model_identifier}: {e}")
                import traceback
                traceback.print_exc()
                decomposer_results.append({
                    "sub_questions": [],
                    "evidence_paths": [],
                    "pipeline_output": {},
                    "final_answer": "",
                    "decomposer": decomp_agent.model_identifier,
                    "error": str(e)
                })

        # 后续 credit 分配、评估等可选：这里只用第一个 decomposer 的结果做后续
        # 如需融合/对比，可遍历 decomposer_results
        
        # 评估每个 decomposer 的结果并给分（简单启发式）
        def f1_score(pred: str, truth: str) -> float:
            if not pred or not truth:
                return 0.0
            pred_tokens = set(pred.lower().split())
            truth_tokens = set(truth.lower().split())
            if not pred_tokens or not truth_tokens:
                return 0.0
            common = pred_tokens & truth_tokens
            if not common:
                return 0.0
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(truth_tokens)
            return 2 * (precision * recall) / (precision + recall)

        def score_decomposer_result(decomp_res: Dict, ground_truth: str) -> Dict:
            sub_questions = decomp_res.get("sub_questions", []) or []
            keywords = decomp_res.get("keywords", []) or []
            evidence_paths = decomp_res.get("evidence_paths", []) or []
            final_answer = decomp_res.get("final_answer", "") or ""

            num_subq_score = min(len(sub_questions) / 4.0, 1.0)
            keyword_score = 0.0
            if keywords and len(sub_questions) > 0:
                keyword_score = min(sum(1 for k in keywords if k and k.strip()) / max(1, len(sub_questions)), 1.0)
            evidence_score = min(len(evidence_paths) / 3.0, 1.0)
            answer_f1 = f1_score(final_answer, ground_truth)
            print("\n evaluatiing on answer: ",final_answer)
            print(f"score: {evidence_score} , f1_s: {answer_f1}")
            
            
            
            score = 0.3 * num_subq_score + 0.2 * keyword_score + 0.2 * evidence_score + 0.3 * answer_f1
            return {
                "decomposer": decomp_res.get("decomposer", "unknown"),
                "score": float(score),
                "components": {
                    "num_subq_score": num_subq_score,
                    "keyword_score": keyword_score,
                    "evidence_score": evidence_score,
                    "answer_f1": answer_f1,
                },
                "final_answer": final_answer,
            }

        decomposer_scores = {}
        per_decomposer_details = []
        best_f1 = 0.0
        any_exact = False
        for dr in decomposer_results:
            detail = score_decomposer_result(dr, task_data['answer'])
            decomposer_scores[detail['decomposer']] = detail
            per_decomposer_details.append(detail)
            best_f1 = max(best_f1, detail['components']['answer_f1'])
            if detail['components']['answer_f1'] == 1.0:
                any_exact = True

        # 使用 scorer_agent 计算每个 decomposer 的 Shapley 值（基于合成答案与 ground truth 的 F1）
        try:
            decomposer_map = {d.get('decomposer', f'd{i}'): d for i, d in enumerate(decomposer_results)}
            use_mc = True if len(decomposer_map) > 6 else False
            shapley_scores = scorer_agent.compute_shapley(decomposer_map, task_data['question'], task_data['answer'], use_mc=use_mc, mc_samples=200)
        except Exception as e:
            print(f"Warning: scorer_agent failed to compute shapley: {e}")
            shapley_scores = {}
            
        print("shapley values:",shapley_scores)
        # compute token deltas for scorer/retriever/verifier and total tokens
        try:
            scorer_end_usage = scorer_agent.llm.get_accumulated_usage() if getattr(scorer_agent, 'llm', None) else {}
        except Exception:
            scorer_end_usage = {}
        try:
            retriever_end_usage = retriever_agent.llm.get_accumulated_usage() if (retriever_agent and getattr(retriever_agent, 'llm', None)) else {}
        except Exception:
            retriever_end_usage = {}
        try:
            verifier_end_usage = verifier_agent.llm.get_accumulated_usage() if (verifier_agent and getattr(verifier_agent, 'llm', None)) else {}
        except Exception:
            verifier_end_usage = {}

        total_decomposer_tokens = sum((d.get('tokens_used', 0.0) for d in decomposer_results))
        total_scorer_tokens = max(0.0, sum_usage(scorer_end_usage) - sum_usage(scorer_start_usage))
        total_retriever_tokens = max(0.0, sum_usage(retriever_end_usage) - sum_usage(retriever_start_usage))
        total_verifier_tokens = max(0.0, sum_usage(verifier_end_usage) - sum_usage(verifier_start_usage))
        total_tokens = total_decomposer_tokens + total_scorer_tokens + total_retriever_tokens + total_verifier_tokens

        # baseline: average F1 across decomposers divided by total tokens (per-sample)
        avg_f1_decomposers = 0.0
        if per_decomposer_details:
            avg_f1_decomposers = sum(d['components']['answer_f1'] for d in per_decomposer_details) / max(1, len(per_decomposer_details))
        baseline = (avg_f1_decomposers / total_tokens) if total_tokens > 0 else 0.0
        print(f"average efficiency: {baseline}\n")

        # compute shapley per token for each decomposer
        for d in decomposer_results:
            did = d.get('decomposer')
            tokens_used = float(d.get('tokens_used', 0.0) or 0.0)
            shap = float(shapley_scores.get(did, 0.0) or 0.0)
            shap_per_token = None if tokens_used <= 0.0 else (shap / tokens_used)
            print(f"{did} score at {shap_per_token} ")
            # update decomposer_scores if present
            if did in decomposer_scores:
                decomposer_scores[did]['components']['tokens_used'] = tokens_used
                decomposer_scores[did]['components']['shapley_per_token'] = shap_per_token
            # also add to per_decomposer_details entries
            for pd in per_decomposer_details:
                if pd.get('decomposer') == did:
                    pd['components']['tokens_used'] = tokens_used
                    pd['components']['shapley_per_token'] = shap_per_token
                    break
 
        sample_result = {
            "sample_id": idx,
            "question": task_data['question'],
            "decomposer_results": decomposer_results,
            "decomposer_scores": decomposer_scores,
            "per_decomposer_details": per_decomposer_details,
            "ground_truth": task_data['answer'],
            "best_f1": best_f1,
            "any_exact": any_exact,
            # Shapley values per decomposer as computed by the scorer agent
            "decomposer_shapley": shapley_scores,
            # baseline metric: average F1 across decomposers divided by total tokens
            "baseline_avgF1_per_token": baseline,
            "total_tokens": total_tokens,
            "total_decomposer_tokens": total_decomposer_tokens,
            "total_scorer_tokens": total_scorer_tokens,
            "total_retriever_tokens": total_retriever_tokens,
            "total_verifier_tokens": total_verifier_tokens,
            # accumulated usage (tokens etc.) observed by the scorer agent's API model
            "accumulated_usage": scorer_agent.llm.get_accumulated_usage() if getattr(scorer_agent, 'llm', None) else {},
            # accumulated usage for each decomposer and for the reasoner (if present)
            "agent_accumulated_usage": {
                **{
                    decomp.model_identifier: (decomp.llm.get_accumulated_usage() if getattr(decomp, 'llm', None) else {})
                    for decomp in decomposer_agents
                },
                **({
                    reasoner_agent.model_identifier: (reasoner_agent.llm.get_accumulated_usage() if getattr(reasoner_agent, 'llm', None) else {})
                } if reasoner_agent else {})
            },
            # scorer synth cache entries
            "scorer_synth_cache": scorer_agent.get_synth_cache_serializable() if hasattr(scorer_agent, 'get_synth_cache_serializable') else [],
        }
        results["samples"].append(sample_result)

        # Save individual result per task_id
        task_result_file = output_dir / f"task_{idx}.json"
        with open(task_result_file, 'w') as f:
            json.dump(sample_result, f, indent=2)
        
        # Update metrics using best decomposer performance for this sample
        results["metrics"]["exact_match"].append(bool(any_exact))
        results["metrics"]["f1_scores"].append(float(best_f1))
        # credit entropy metrics aren't computed here; append 0.0 placeholders
        results["metrics"]["credit_entropy_static"].append(0.0)
        if use_dynamic_credit:
            results["metrics"]["credit_entropy_dynamic"].append(0.0)

    # Calculate aggregate metrics
    em_accuracy = sum(results["metrics"]["exact_match"]) / len(results["metrics"]["exact_match"])
    avg_f1 = sum(results["metrics"]["f1_scores"]) / len(results["metrics"]["f1_scores"])
    avg_static_entropy = sum(results["metrics"]["credit_entropy_static"]) / len(results["metrics"]["credit_entropy_static"])

    results["aggregate"] = {
        "exact_match_accuracy": em_accuracy,
        "average_f1": avg_f1,
        "average_static_entropy": avg_static_entropy,
    }

    if use_dynamic_credit:
        avg_dynamic_entropy = sum(results["metrics"]["credit_entropy_dynamic"]) / len(results["metrics"]["credit_entropy_dynamic"])
        results["aggregate"]["average_dynamic_entropy"] = avg_dynamic_entropy

    # Save results
    results_file = output_dir / "hotpotqa_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"Exact Match Accuracy: {em_accuracy:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Average Static Entropy: {avg_static_entropy:.3f}")
    if use_dynamic_credit:
        print(f"Average Dynamic Entropy: {avg_dynamic_entropy:.3f}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")

    return results


def main():
    """Main function."""
    # Example: pass a dict mapping agent ids / roles to model identifiers.
    # You can specify models for main agents by "agent_0", "agent_1", ...
    # and for special agents by "verifier", "reasoner", "decomposer".
    agent_models = {
        "agent_0": "Qwen/Qwen2.5-7B-Instruct",
        "agent_1": "deepseek-ai/DeepSeek-V3",
        "agent_2": "openai/gpt-5.1",
        "verifier": "openai/gpt-5.1",
        "reasoner": "deepseek-ai/DeepSeek-V3",
        "decomposer_0": "openai/gpt-5.1",
        "decomposer_1": "deepseek-ai/DeepSeek-V3",
        "decomposer_2": "Qwen/Qwen2.5-7B-Instruct"
    }

    results = run_hotpotqa_experiment(
        data_path="data/hotpot_dev_fullwiki_v1.json",
        max_samples=3,  # Increased from 10 to 20
        num_agents=3,
        model_identifier="Qwen/Qwen2.5-7B-Instruct",  # Default model
        use_decomposer=True,
        use_rag=True,  # Enable RAG pipeline
        use_dynamic_credit=True,
        agent_model_identifiers=agent_models,
        output_dir=Path("results/hotpotqa_rag_fix")  # New output directory
    )
    
    # pprint.pprint(results)


if __name__ == "__main__":
    main()