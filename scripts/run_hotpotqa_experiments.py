"""Experiment runner for HotpotQA with dynamic credit allocation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment_hotpotqa import HotpotQAEnvironment
from llm_agent import LLMAgent
from decomposer_agent import DecomposerAgent
from retriever_agent import RetrieverAgent
from evidence_verifier_agent import EvidenceVerifierAgent
from controller import MultiAgentController
from reward_manager import RewardManager


def create_agents_with_rag(
    num_agents: int,
    model_identifier: str,
    agent_roles: List[str] = None,
    use_rag: bool = True,
    corpus_path: str = None,
    use_v3: bool = False
) -> tuple[List[LLMAgent], DecomposerAgent]:
    """Create agents including decomposer, retriever, and verifier.

    Parameters
    ----------
    num_agents : int
        Number of main agents to create.
    model_identifier : str
        Model identifier for LLM API.
    agent_roles : List[str]
        List of roles for agents.
    use_rag : bool
        Whether to include RAG agents (retriever and verifier).
    corpus_path : str
        Path to corpus for retrieval (optional).
    use_v3 : bool
        Whether to use v3 enhanced agents (Retriever v3, Reasoner v3).

    Returns
    -------
    tuple[List[LLMAgent], DecomposerAgent]
        Main agents and decomposer agent.
    """
    if agent_roles is None:
        default_roles = ["planner", "coder", "reviewer", "tester"]
        agent_roles = (default_roles * ((num_agents // len(default_roles)) + 1))[:num_agents]

    # Create main agents
    agents = []
    for i in range(num_agents):
        role = agent_roles[i] if i < len(agent_roles) else "assistant"
        system_prompt = f"You are a {role}. Your goal is to help answer multi-hop questions by reasoning step by step."

        agent = LLMAgent(
            agent_id=f"agent_{i}",
            model_identifier=model_identifier,
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

            # Create evidence verifier agent
            verifier = EvidenceVerifierAgent(
                agent_id="verifier",
                model_identifier=model_identifier,
                max_tokens=256,
                temperature=0.7,
                min_entity_overlap=0.5,  # Added parameter
                use_spacy=True  # Enable spaCy if available
            )
            agents.append(verifier)
            print("✓ Added EvidenceVerifierAgent")

            # Create reasoner agent for answer synthesis (v3 or regular)
            if use_v3:
                from reasoner_agent_v3 import ReasonerAgentV3
                reasoner = ReasonerAgentV3(
                    agent_id="reasoner",
                    model_identifier=model_identifier,
                    max_tokens=1024,
                    temperature=0.3  # Lower temperature for more consistent answers
                )
                print("✓ Added ReasonerAgentV3 (enhanced)")
            else:
                from reasoner_agent import ReasonerAgent
                reasoner = ReasonerAgent(
                    agent_id="reasoner",
                    model_identifier=model_identifier,
                    max_tokens=512,
                    temperature=0.7
                )
                print("✓ Added ReasonerAgent")
            agents.append(reasoner)

        except Exception as e:
            print(f"⚠ Could not create RAG agents: {e}")
            print("Continuing without RAG components...")

    # Create decomposer agent
    decomposer = DecomposerAgent(
        agent_id="decomposer",
        model_identifier=model_identifier,
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
    use_v3: bool = False,
    use_iterative: bool = False,  # New parameter for iterative RAG
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
        Model identifier for LLM API.
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
    print(f"  Model: {model_identifier}")
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
    agents, decomposer = create_agents_with_rag(
        num_agents=num_agents,
        model_identifier=model_identifier,
        use_rag=use_rag,
        use_v3=use_v3
    )

    # Initialize reward manager
    reward_manager = RewardManager(base_reward=1.0)

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
    for idx in range(min(max_samples, env.get_num_samples())):
        print(f"\n--- Sample {idx + 1}/{max_samples} ---")

        # Set current task
        env.set_current_task(idx)
        task_data = env.current_task_data

        print(f"Question: {task_data['question']}")
        print(f"Answer: {task_data['answer']}")

        # Create controller
        controller = MultiAgentController(
            agents=agents,
            environment=env,
            mode="sequential",
            use_decomposer=use_decomposer,
            decomposer_agent=decomposer if use_decomposer else None
        )

        # Run HotpotQA RAG pipeline if RAG is enabled
        sub_questions = []
        evidence_paths = []
        pipeline_output = {}
        final_answer = ""

        if use_rag:
            try:
                # Use iterative RAG pipeline if enabled and v3 is active
                if use_iterative and use_v3:
                    print("Using Iterative RAG Pipeline...")
                    from iterative_rag_pipeline import IterativeRAGPipeline

                    # Find the required agents
                    decomp_agent = decomposer
                    retriever_agent = None
                    verifier_agent = None
                    reasoner_agent = None

                    for agent in agents:
                        if hasattr(agent, 'role'):
                            if agent.role == 'retriever':
                                retriever_agent = agent
                            elif agent.role in ['verifier', 'evidence_verifier']:  # Support both role names
                                verifier_agent = agent
                            elif agent.role == 'reasoner':
                                reasoner_agent = agent

                    if all([decomp_agent, retriever_agent, verifier_agent, reasoner_agent]):
                        # Create iterative pipeline
                        pipeline = IterativeRAGPipeline(
                            decomposer=decomp_agent,
                            retriever=retriever_agent,
                            verifier=verifier_agent,
                            reasoner=reasoner_agent,
                            use_iterative=True
                        )

                        # Run pipeline
                        pipeline_output = pipeline.run(task_data['question'])
                        sub_questions = pipeline_output.get("sub_questions", [])
                        final_answer = pipeline_output.get("final_answer", "unknown")

                        # Extract evidence paths
                        agent_outputs_rag = pipeline_output.get("agent_outputs", {})
                        for sq, output in agent_outputs_rag.items():
                            if "evidence_path" in output:
                                evidence_paths.extend(output["evidence_path"])

                        print(f"Sub-questions: {len(sub_questions)}")
                        print(f"Evidence retrieved: {len(evidence_paths)} passages")
                        print(f"Iterative answers: {len(pipeline_output.get('intermediate_answers', {}))}")
                        print(f"Final answer: {final_answer[:100]}...")
                    else:
                        print("⚠ Missing required agents for iterative pipeline, falling back to standard RAG")
                        use_iterative = False

                # Standard RAG pipeline (non-iterative)
                if not use_iterative:
                    import asyncio
                    task = {"question": task_data['question']}
                    # Run async pipeline
                    pipeline_output = asyncio.run(controller.run_hotpotqa_pipeline(task))
                sub_questions = pipeline_output.get("sub_questions", [])

                # Extract evidence paths
                agent_outputs_rag = pipeline_output.get("agent_outputs", {})
                for sq, output in agent_outputs_rag.items():
                    if "evidence_path" in output:
                        evidence_paths.extend(output["evidence_path"])

                print(f"Sub-questions: {len(sub_questions)}")
                print(f"Evidence retrieved: {len(evidence_paths)} passages")

                # Use ReasonerAgent to generate final answer from RAG results
                reasoner = None
                for agent in agents:
                    if hasattr(agent, 'role') and agent.role == 'reasoner':
                        reasoner = agent
                        break

                if reasoner:
                    print("Using ReasonerAgent to synthesize answer...")
                    reasoner_result = reasoner.synthesize_answer(
                        main_question=task_data['question'],
                        sub_results=agent_outputs_rag
                    )
                    final_answer = reasoner_result['final_answer']
                    print(f"ReasonerAgent answer: {final_answer[:100]}...")
                else:
                    print("⚠ ReasonerAgent not found, using standard execution...")
                    # Fallback to standard execution
                    task_result = controller.run_sync()
                    agent_outputs = task_result.agent_responses
                    final_answer = list(agent_outputs.values())[-1] if agent_outputs else ""

            except Exception as e:
                print(f"Warning: RAG pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to standard execution...")
                # Fallback to standard execution
                task_result = controller.run_sync()
                agent_outputs = task_result.agent_responses
                final_answer = list(agent_outputs.values())[-1] if agent_outputs else ""
        else:
            # Run standard task execution (no RAG)
            task_result = controller.run_sync()
            agent_outputs = task_result.agent_responses
            final_answer = list(agent_outputs.values())[-1] if agent_outputs else ""

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
        contributions = {
            agent.agent_id: len(agent_outputs.get(agent.agent_id, "").split()) / 100.0
            for agent in agents
        }
        static_rewards = reward_manager.allocate_shapley_rewards(
            agents=agents,
            contributions=contributions,
            use_monte_carlo=True,
            num_samples=100
        )

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
            "sub_questions": sub_questions,  # Added per workflow
            "evidence_paths": evidence_paths,  # Added per workflow
            "ground_truth": ground_truth,
            "predicted_answer": final_answer,
            "exact_match": task_result.metadata.get('exact_match', False),
            "f1_score": task_result.metadata.get('f1_score', 0.0),
            "static_rewards": static_rewards,
            "dynamic_credits": dynamic_credits,  # Added per workflow (credit tracking)
            "static_entropy": static_entropy,
            "dynamic_entropy": dynamic_entropy,
            "pipeline_output": pipeline_output,  # Complete pipeline output
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
    results = run_hotpotqa_experiment(
        data_path="data/hotpot_dev_fullwiki_v1.json",
        max_samples=20,  # Increased from 10 to 20
        num_agents=3,
        model_identifier="Qwen/Qwen2.5-7B-Instruct",  # Changed to Qwen for faster results
        use_decomposer=True,
        use_rag=True,  # Enable RAG pipeline
        use_dynamic_credit=True,
        output_dir=Path("results/hotpotqa_rag_fix")  # New output directory
    )


if __name__ == "__main__":
    main()
