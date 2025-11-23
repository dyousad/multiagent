"""Experiment runner for HotpotQA with dynamic credit allocation."""

from __future__ import annotations

import json
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
            temperature=0.7
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
                    max_tokens=512,
                    temperature=0.7
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
        temperature=0.7
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
                temperature=0.7
            )
        )

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
                elif agent.role == 'reasoner':
                    reasoner_agent = agent

        # 多 decomposer 分别调用 pipeline
        decomposer_results = {}
        for decomp_agent in decomposer_agents:
            print(f"\n>>> Using decomposer: {decomp_agent.model_identifier}")
            try:
                decomposition = decomp_agent.decompose_question(task_data['question'])
                sub_questions = decomposition.get("sub_questions", [])
                evidence_paths = []
                pipeline_output = {}
                final_answer = ""

                if use_rag and use_iterative and use_v3 and all([retriever_agent, verifier_agent, reasoner_agent]):
                    from iterative_rag_pipeline import IterativeRAGPipeline
                    pipeline = IterativeRAGPipeline(
                        sub_questions=sub_questions,
                        retriever=retriever_agent,
                        verifier=verifier_agent,
                        reasoner=reasoner_agent,
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

                # 结果收集
                decomposer_results[decomp_agent.model_identifier] = {
                    "sub_questions": sub_questions,
                    "evidence_paths": evidence_paths,
                    "pipeline_output": pipeline_output,
                    "final_answer": final_answer
                }
            except Exception as e:
                print(f"Warning: pipeline failed for decomposer {decomp_agent.model_identifier}: {e}")
                import traceback
                traceback.print_exc()
                decomposer_results[decomp_agent.model_identifier] = {
                    "error": str(e)
                }

        # 后续 credit 分配、评估等可选：这里只用第一个 decomposer 的结果做后续
        # 如需融合/对比，可遍历 decomposer_results
        first_result = next(iter(decomposer_results.values()))
        sub_questions = first_result.get("sub_questions", [])
        evidence_paths = first_result.get("evidence_paths", [])
        pipeline_output = first_result.get("pipeline_output", {})
        final_answer = first_result.get("final_answer", "")
        agent_outputs_rag = pipeline_output.get("agent_outputs", {})
        # ...existing code...

        # For evaluation, still need agent_outputs for credit allocation
        if use_rag and final_answer and final_answer not in ["", "Error: No question provided"]:
            # Already have answer from RAG+Reasoner
            # Create agent_outputs from RAG pipeline results
            agent_outputs = {}
            for sq, result in agent_outputs_rag.items():
                # Include evidence in outputs for credit calculation
                if isinstance(result, dict) and 'evidence' in result:
                    agent_outputs[f"rag_{sq[:20]}"] = " ".join(result['evidence'][:2])

            # Add reasoner output
            agent_outputs['reasoner'] = final_answer

            # Calculate exact match and F1 score
            ground_truth = task_data['answer']
            pred_normalized = final_answer.lower().strip()
            truth_normalized = ground_truth.lower().strip()
            exact_match = (pred_normalized == truth_normalized)

            # Calculate F1 score
            pred_tokens = set(final_answer.lower().split())
            truth_tokens = set(ground_truth.lower().split())
            if len(pred_tokens) == 0 or len(truth_tokens) == 0:
                f1_score = 0.0
            else:
                common_tokens = pred_tokens & truth_tokens
                if len(common_tokens) == 0:
                    f1_score = 0.0
                else:
                    precision = len(common_tokens) / len(pred_tokens)
                    recall = len(common_tokens) / len(truth_tokens)
                    f1_score = 2 * (precision * recall) / (precision + recall)

            # Create a properly initialized TaskResult
            from environment import TaskResult
            task_result = TaskResult(
                task_description=task_data['question'],
                agent_responses=agent_outputs,
                final_output=final_answer,
                success=True,
                metadata={
                    "exact_match": exact_match,
                    "f1_score": f1_score,
                    "ground_truth": ground_truth,
                    "predicted_answer": final_answer
                }
            )
        else:
            # Need to run standard execution or RAG failed
            if not (use_rag and pipeline_output):
                task_result = controller.run_sync()
                agent_outputs = task_result.agent_responses
                final_answer = list(agent_outputs.values())[-1] if agent_outputs else ""
            else:
                # RAG ran but no valid answer, still get agent_outputs from standard execution
                task_result = controller.run_sync()
                agent_outputs = task_result.agent_responses
                if not final_answer or final_answer in ["", "Error: No question provided"]:
                    final_answer = list(agent_outputs.values())[-1] if agent_outputs else ""
        ground_truth = task_data['answer']

        print(f"Predicted: {final_answer[:100]}...")
        print(f"Exact Match: {task_result.metadata.get('exact_match', False)}")
        print(f"F1 Score: {task_result.metadata.get('f1_score', 0.0):.3f}")

        # Static credit allocation (Shapley)
        
        def calculate_meaningful_contributions(agent_outputs, final_answer, ground_truth, task_data):
            """Calculate contributions based on actual impact on task performance."""
            contributions = {}
            
            for agent_id, output in agent_outputs.items():
                # Start with baseline contribution
                score = 0.1  # Base minimum contribution for participation
                
                # 1. Semantic relevance to final answer
                if final_answer and output:
                    answer_similarity = calculate_semantic_similarity(output, final_answer)
                    score += answer_similarity * 0.3
                
                # 2. Evidence quality (for RAG agents)
                if "evidence" in str(output).lower() or "retrieve" in agent_id.lower():
                    evidence_score = calculate_evidence_relevance(output, task_data.get('context', []))
                    score += evidence_score * 0.3
                
                # 3. Reasoning quality
                reasoning_score = assess_reasoning_quality(output, task_data['question'])
                score += reasoning_score * 0.3
                
                # 4. Role-based baseline contributions
                if "reason" in agent_id.lower():
                    score += 0.2  # Reasoner gets extra baseline
                elif "retrieve" in agent_id.lower():
                    score += 0.15  # Retriever baseline
                elif "verif" in agent_id.lower():
                    score += 0.1  # Verifier baseline
                
                contributions[agent_id] = min(score, 1.0)  # Cap at 1.0
            
            return contributions
        
        def calculate_semantic_similarity(text1, text2):
            """Calculate semantic similarity between two texts."""
            if not text1 or not text2:
                return 0.0
            
            # Simple word overlap as fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))

        def calculate_evidence_relevance(output, context):
            """Calculate how relevant the evidence is to the context."""
            if not context:
                return 0.0
            
            # Count named entities that match context
            try:
                doc = nlp(output)
                context_doc = nlp(" ".join(context))
                
                context_entities = set(ent.text.lower() for ent in context_doc.ents)
                output_entities = set(ent.text.lower() for ent in doc.ents)
                
                if not context_entities:
                    return 0.0
                    
                overlap = len(output_entities.intersection(context_entities))
                return overlap / len(context_entities)
            except:
                return 0.0

        def assess_reasoning_quality(output, question):
            """Assess the quality of reasoning in the output."""
            reasoning_indicators = [
                "because", "therefore", "thus", "since", "as a result",
                "first", "second", "then", "finally", "conclusion"
            ]
            
            output_lower = output.lower()
            indicator_count = sum(1 for indicator in reasoning_indicators 
                                if indicator in output_lower)
            
            # Normalize by text length
            word_count = len(output.split())
            if word_count == 0:
                return 0.0
            
            return min(indicator_count / (word_count / 50), 1.0)  # Cap at 1.0
        
        # Enhanced value function that models synergy
        def create_synergistic_value_function(contributions, task_data, final_answer, ground_truth):
            """Create a value function that captures agent synergy."""
            
            def value_function(coalition: Set[str]) -> float:
                if not coalition:
                    return 0.0
                
                # Base value: sum of individual contributions
                base_value = sum(contributions.get(agent, 0.0) for agent in coalition)
                
                # Synergy bonus: agents working together create more value
                synergy_bonus = 0.0
                
                # Check for complementary agent types
                agent_types = [agent.split('_')[0] for agent in coalition]  # Extract agent type prefix
                
                # Bonus for having both retrieval and reasoning agents
                if any('retrieve' in agent for agent in coalition) and any('reason' in agent for agent in coalition):
                    synergy_bonus += 0.3
                
                # Bonus for verifier presence
                if any('verif' in agent for agent in coalition):
                    synergy_bonus += 0.2
                
                # Penalty for missing critical roles in complex questions
                question_complexity = len(task_data.get('question', '').split())
                if question_complexity > 10 and any('decompose' in agent for agent in coalition):
                    synergy_bonus += 0.2
                
                # Scale synergy by coalition size (diminishing returns)
                synergy_bonus *= max(0, (1 - 0.05 * len(coalition)))  # Much gentler scaling
                
                return min(base_value * (1 + synergy_bonus), 1.0)  # Cap at 1.0
            
            return value_function
        
        def validate_shapley_values(shapley_values, contributions, agent_outputs):
            """Validate that Shapley values make sense."""
            print("\n--- Shapley Value Validation ---")
            
            total_shapley = sum(shapley_values.values())
            print(f"Total Shapley value: {total_shapley:.3f}")
            
            for agent_id, shapley_val in shapley_values.items():
                contribution = contributions.get(agent_id, 0.0)
                output_preview = agent_outputs.get(agent_id, "")[:100] + "..." if agent_outputs.get(agent_id) else "None"
                
                print(f"{agent_id}:")
                print(f"  Shapley: {shapley_val:.3f}")
                print(f"  Contribution: {contribution:.3f}")
                print(f"  Output: {output_preview}")
                
                # Check for anomalies
                if shapley_val > 1.0:
                    print(f"  ⚠ High Shapley value!")
                if shapley_val < 0:
                    print(f"  ⚠ Negative Shapley value!")

        contributions = calculate_meaningful_contributions(
            agent_outputs=agent_outputs,
            final_answer=final_answer,
            ground_truth=ground_truth,
            task_data=task_data
        ) 
        
        value_function = create_synergistic_value_function(
            contributions=contributions,
            task_data=task_data,
            final_answer=final_answer,
            ground_truth=ground_truth
        )

        
        static_rewards = reward_manager.allocate_shapley_rewards(
            agents=agents,
            contributions=contributions,  # Still needed for fallback
            value_function=value_function,  # Use our improved function
            use_monte_carlo=True,
            num_samples=200  # Increased samples for better accuracy
        )

        # validate_shapley_values(static_rewards, contributions, agent_outputs)
        
        # Calculate static credit entropy
        static_entropy = reward_manager.get_credit_entropy(static_rewards)

        # Dynamic credit allocation (only for valid answers)
        dynamic_credits = {}
        dynamic_entropy = 0.0
        if use_dynamic_credit:
            # Check if answer is valid before updating credit
            if final_answer and final_answer not in ["", "Error: No question provided", "No evidence found."]:
                # Define evaluation function
                def f1_eval(pred: str, truth: str) -> float:
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

                dynamic_credits = reward_manager.update_credits_dynamic(
                    agent_outputs=agent_outputs,
                    final_answer=final_answer,
                    ground_truth=ground_truth,
                    evaluate_fn=f1_eval
                )
                dynamic_entropy = reward_manager.get_credit_entropy(dynamic_credits)

                print(f"Dynamic Credits: {dynamic_credits}")
            else:
                print(f"⚠ Skipping credit update: invalid final answer")

        # Record results with enhanced logging (as per workflow spec)

        sample_result = {
            "sample_id": idx,
            "question": task_data['question'],
            "decomposer_results": decomposer_results,  # 保存所有 decomposer 的结果
            "sub_questions": sub_questions,
            "evidence_paths": evidence_paths,
            "ground_truth": ground_truth,
            "predicted_answer": final_answer,
            "exact_match": task_result.metadata.get('exact_match', False),
            "f1_score": task_result.metadata.get('f1_score', 0.0),
            "static_rewards": static_rewards,
            "dynamic_credits": dynamic_credits,
            "static_entropy": static_entropy,
            "dynamic_entropy": dynamic_entropy,
            "pipeline_output": pipeline_output,
        }
        results["samples"].append(sample_result)

        # Save individual result per task_id (as per workflow spec)
        task_result_file = output_dir / f"task_{idx}.json"
        with open(task_result_file, 'w') as f:
            json.dump(sample_result, f, indent=2)

        # Update metrics
        results["metrics"]["exact_match"].append(task_result.metadata.get('exact_match', False))
        results["metrics"]["f1_scores"].append(task_result.metadata.get('f1_score', 0.0))
        results["metrics"]["credit_entropy_static"].append(static_entropy)
        if use_dynamic_credit:
            results["metrics"]["credit_entropy_dynamic"].append(dynamic_entropy)

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
        "agent_2": "openai/gpt-4-turbo",
        "verifier": "openai/gpt-4-turbo",
        "reasoner": "deepseek-ai/DeepSeek-V3",
        "decomposer_0": "Qwen/Qwen2.5-7B-Instruct",
        "decomposer_1": "deepseek-ai/DeepSeek-V3",
        "decomposer_2": "openai/gpt-4-turbo"
    }

    results = run_hotpotqa_experiment(
        data_path="data/hotpot_dev_fullwiki_v1.json",
        max_samples=2,  # Increased from 10 to 20
        num_agents=3,
        model_identifier="Qwen/Qwen2.5-7B-Instruct",  # Default model
        use_decomposer=True,
        use_rag=True,  # Enable RAG pipeline
        use_dynamic_credit=True,
        agent_model_identifiers=agent_models,
        output_dir=Path("results/hotpotqa_rag_fix")  # New output directory
    )


if __name__ == "__main__":
    main()