"""Multi-agent controller for parallel execution."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import itertools

from agent import Agent, Message
from environment import Environment, TaskResult


class MultiAgentController:
    """Controller for managing multiple agents in parallel.

    The controller:
    - Launches multiple agents
    - Coordinates turn-based or message-passing interactions
    - Aggregates agent outputs
    - Manages the execution flow
    - Supports question decomposition for complex multi-hop tasks
    """

    def __init__(
        self,
        agents: List[Agent],
        environment: Environment,
        mode: str = "parallel",
        use_decomposer: bool = False,
        decomposer_agent: Optional[Agent] = None,
    ):
        """Initialize the multi-agent controller.

        Parameters
        ----------
        agents : List[Agent]
            List of agents to coordinate.
        environment : Environment
            The task environment.
        mode : str
            Execution mode: "parallel" (all agents act simultaneously),
            "sequential" (agents take turns), or "message_passing" (agents communicate).
        use_decomposer : bool
            Whether to use a decomposer agent for question decomposition.
        decomposer_agent : Optional[Agent]
            The decomposer agent to use (if use_decomposer is True).
        """
        self.agents = agents
        self.environment = environment
        self.mode = mode
        self.message_queue: List[Message] = []
        self.use_decomposer = use_decomposer
        self.decomposer_agent = decomposer_agent
        self.sub_questions: List[str] = []

    async def run_parallel(self) -> TaskResult:
        """Run all agents in parallel for one round.

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        final_responses = {}

        while self.environment.next_round() or self.environment.current_round == 0:
            # Get observation for all agents (empty previous responses in parallel mode)
            tasks = []
            for agent in self.agents:
                obs = self.environment.get_observation(agent.agent_id)
                # Create async task for each agent
                tasks.append(self._run_agent_async(agent, obs))

            # Execute all agents in parallel
            responses = await asyncio.gather(*tasks)

            # Record responses
            for agent, response in zip(self.agents, responses):
                self.environment.record_response(agent.agent_id, response)
                final_responses[agent.agent_id] = response

            # In parallel mode, we typically run just one round
            if self.mode == "parallel":
                break

        return self.environment.evaluate(final_responses)

    async def run_sequential(self) -> TaskResult:
        """Run agents sequentially, each seeing previous responses.

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        final_responses = {}

        # Step 1: Decompose question if decomposer is enabled
        if self.use_decomposer and self.decomposer_agent:
            print("Decomposing question with decomposer agent...")
            decomposer_obs = self.environment.get_observation("decomposer")
            decomposition_response = await self._run_agent_async(
                self.decomposer_agent, decomposer_obs
            )
            self.environment.record_response("decomposer", decomposition_response)
            final_responses["decomposer"] = decomposition_response

            # Parse sub-questions if decomposer has this method
            if hasattr(self.decomposer_agent, 'parse_sub_questions'):
                self.sub_questions = self.decomposer_agent.parse_sub_questions(
                    decomposition_response
                )
                print(f"Generated {len(self.sub_questions)} sub-questions")

        # Step 2: Run main agents sequentially
        while self.environment.next_round() or self.environment.current_round == 0:
            round_responses = []

            # Include sub-questions in context if available
            if self.sub_questions:
                sub_q_context = "Sub-questions to address:\n" + "\n".join(
                    f"{i+1}. {sq}" for i, sq in enumerate(self.sub_questions)
                )
                round_responses.append(f"Decomposer: {sub_q_context}")

            for agent in self.agents:
                # Each agent sees previous responses in this round
                obs = self.environment.get_observation(
                    agent.agent_id,
                    previous_responses=round_responses
                )
                response = await self._run_agent_async(agent, obs)
                round_responses.append(f"{agent.agent_id}: {response}")
                self.environment.record_response(agent.agent_id, response)
                final_responses[agent.agent_id] = response

            # Check if task is complete (simple heuristic)
            if self.environment.current_round >= self.environment.max_rounds - 1:
                break

        return self.environment.evaluate(final_responses)

    async def run_message_passing(self, max_iterations: int = 10) -> TaskResult:
        """Run agents with message passing communication.

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        final_responses = {}

        # Initial round: all agents generate initial responses
        tasks = []
        for agent in self.agents:
            obs = self.environment.get_observation(agent.agent_id)
            tasks.append(self._run_agent_async(agent, obs))

        responses = await asyncio.gather(*tasks)
        for agent, response in zip(self.agents, responses):
            self.environment.record_response(agent.agent_id, response)
            final_responses[agent.agent_id] = response

        # Message passing rounds
        for iteration in range(max_iterations):
            # Deliver queued messages
            for msg in self.message_queue:
                for agent in self.agents:
                    agent.receive(msg)
            self.message_queue.clear()

            # Agents can communicate (simplified: broadcast)
            # In practice, you'd implement more sophisticated message routing
            if iteration < max_iterations - 1:
                for agent in self.agents:
                    # Agent decides whether to send a message
                    # (This is a placeholder - in practice, agent.act() would handle this)
                    pass

        return self.environment.evaluate(final_responses)

    async def _run_agent_async(self, agent: Agent, observation: Dict[str, Any]) -> str:
        """Run a single agent asynchronously.

        Parameters
        ----------
        agent : Agent
            The agent to run.
        observation : Dict[str, Any]
            Current observation.

        Returns
        -------
        str
            The agent's response.
        """
        # Wrap synchronous agent.act() in asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.act, observation)
        return response

    async def run(self) -> TaskResult:
        """Run the multi-agent system according to the configured mode.

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        if self.mode == "parallel":
            return await self.run_parallel()
        elif self.mode == "sequential":
            return await self.run_sequential()
        elif self.mode == "message_passing":
            return await self.run_message_passing()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    async def run_hotpotqa_pipeline(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run HotpotQA RAG pipeline with decomposition, retrieval, and verification.

        This pipeline:
        1. Decomposes the question into sub-questions (if decomposer is available)
        2. For each sub-question:
           - Retrieves evidence using the retriever agent
           - Verifies evidence quality using the verifier agent
           - Stores results for downstream processing
        3. Returns structured output with all agent contributions

        Parameters
        ----------
        task : Dict[str, Any]
            Task containing the question to answer.
            Expected to have 'question' or 'task' key.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'sub_questions': List of sub-questions (if decomposer used)
            - 'agent_outputs': Dict mapping sub-question to retrieval/verification results
            - 'decomposition': Original decomposition response (if decomposer used)
        """
        agent_outputs = {}
        sub_questions = []

        # Helper: simple semantic similarity (word-overlap)
        def semantic_similarity(text1: str, text2: str) -> float:
            if not text1 or not text2:
                return 0.0
            w1 = set(text1.lower().split())
            w2 = set(text2.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / max(len(w1), len(w2))

        # Helper: compute Shapley values for a set of agents' outputs
        def compute_shapley_for_answers(agent_answers: Dict[str, str], reference: str) -> Dict[str, float]:
            # value function: similarity between concatenated coalition answers and reference
            agents = list(agent_answers.keys())

            def v(coalition: List[str]) -> float:
                if not coalition:
                    return 0.0
                combined = " ".join(agent_answers[a] for a in coalition if a in agent_answers)
                return semantic_similarity(combined, reference)

            n = len(agents)
            shapley: Dict[str, float] = {a: 0.0 for a in agents}

            # Enumerate all coalitions
            import math
            for a in agents:
                for r in range(0, n):
                    for subset in itertools.combinations([x for x in agents if x != a], r):
                        subset = list(subset)
                        subset_with = subset + [a]
                        weight = (float(math.factorial(r) * math.factorial(n - r - 1)) / float(math.factorial(n))) if n > 0 else 0.0
                        marginal = v(subset_with) - v(subset)
                        shapley[a] += weight * marginal

            return shapley

        # Step 1: Decompose question if decomposer is available
        if self.use_decomposer and self.decomposer_agent:
            print("Step 1: Decomposing question...")
            decomposer_obs = {"task": task.get("question", task.get("task", ""))}
            decomposition_response = await self._run_agent_async(
                self.decomposer_agent, decomposer_obs
            )

            # Parse sub-questions if available
            if hasattr(self.decomposer_agent, "parse_sub_questions"):
                sub_questions = self.decomposer_agent.parse_sub_questions(
                    decomposition_response
                )
            else:
                # Fallback: use original question as single sub-question
                sub_questions = [task.get("question", task.get("task", ""))]

            print(f"Generated {len(sub_questions)} sub-questions")
            agent_outputs["decomposition"] = decomposition_response

        else:
            # No decomposer: use original question
            sub_questions = [task.get("question", task.get("task", ""))]

        # Step 2: Process each sub-question with retriever and verifier
        retrieval_outputs = {}
        for i, sub_q in enumerate(sub_questions, 1):
            print(f"\nStep 2.{i}: Processing sub-question: {sub_q[:80]}...")

            sq_outputs = {}

            # 2a. Retrieve evidence
            retriever = self._get_agent_by_role("retriever")
            if retriever:
                print(f"  - Retrieving evidence...")
                retrieval_obs = {"sub_question": sub_q}

                if hasattr(retriever, "retrieve_evidence"):
                    # Use structured retrieval method
                    retrieval_result = retriever.retrieve_evidence(sub_q)
                    sq_outputs["evidence"] = retrieval_result["evidence"]
                    sq_outputs["evidence_path"] = retrieval_result["path"]
                else:
                    # Use generic act method
                    retrieval_response = await self._run_agent_async(
                        retriever, retrieval_obs
                    )
                    sq_outputs["retrieval_response"] = retrieval_response
                    sq_outputs["evidence"] = []  # Couldn't extract structured evidence

            # 2b. Verify evidence
            verifier = self._get_agent_by_role("evidence_verifier")
            if verifier and "evidence" in sq_outputs:
                print(f"  - Verifying evidence...")
                verify_obs = {
                    "sub_question": sub_q,
                    "evidence": sq_outputs["evidence"],
                }

                if hasattr(verifier, "verify_evidence"):
                    # Use structured verification method
                    verify_result = verifier.verify_evidence(
                        sub_q, sq_outputs["evidence"]
                    )
                    sq_outputs.update(verify_result)
                else:
                    # Use generic act method
                    verify_response = await self._run_agent_async(verifier, verify_obs)
                    sq_outputs["verification_response"] = verify_response

            retrieval_outputs[sub_q] = sq_outputs

        return {
            "sub_questions": sub_questions,
            "agent_outputs": retrieval_outputs,
            "decomposition": agent_outputs.get("decomposition", ""),
        }

    def _get_agent_by_role(self, role: str) -> Optional[Agent]:
        """Get an agent by role.

        Parameters
        ----------
        role : str
            The role to search for.

        Returns
        -------
        Optional[Agent]
            The agent with the matching role, or None if not found.
        """
        for agent in self.agents:
            if hasattr(agent, "role") and agent.role == role:
                return agent
        return None

    def run_sync(self) -> TaskResult:
        """Synchronous wrapper for run().

        Returns
        -------
        TaskResult
            Result of the task execution.
        """
        return asyncio.run(self.run())
