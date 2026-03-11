"""Hierarchical Planner Agent — decomposes complex queries into a DAG of sub-tasks.

The planner sits above the CustomerInsightAgent. It:
1. Analyzes the user query and produces a plan (a DAG of sub-tasks)
2. Executes independent sub-tasks in parallel via asyncio
3. Synthesizes sub-agent results into a final answer
4. Can escalate to a human when confidence is low

The plan is represented as a list of PlanStep objects with dependency edges,
enabling topological-sort-based parallel execution.
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
from google import genai
from google.genai import types

from .agent import CustomerInsightAgent, AgentResponse
from .critic import CriticAgent
from ..api_utils import generate_with_retry, DEFAULT_MODEL


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class PlanStep:
    """A single step in the execution plan (node in the DAG)."""

    id: str
    task: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    agent_response: Optional[AgentResponse] = None


@dataclass
class ExecutionPlan:
    """A DAG of sub-tasks produced by the planner."""

    query: str
    steps: list[PlanStep] = field(default_factory=list)
    synthesis_instruction: str = ""
    should_escalate: bool = False
    escalation_reason: str = ""


@dataclass
class PlannerResponse:
    """Complete response from the hierarchical planner."""

    answer: str
    plan: ExecutionPlan
    sub_results: dict[str, str] = field(default_factory=dict)
    escalated: bool = False
    escalation_reason: str = ""


# ── Prompts ──────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a Planning Agent that decomposes complex analytical queries
into smaller, independent sub-tasks. Each sub-task will be handled by a specialist agent
with access to search, extraction, sentiment analysis, and summarization tools.

USER QUERY: {query}

Analyze the query and produce a JSON execution plan. Rules:
- Break the query into 2-5 sub-tasks that can be executed by specialist agents
- Identify dependencies: if step B needs results from step A, mark it
- Steps with no dependencies on each other can run IN PARALLEL
- If the query is simple enough for one agent, return a single step
- If the query is ambiguous, sensitive, or you lack confidence, set should_escalate=true

Return this exact JSON structure:
{{
    "steps": [
        {{
            "id": "step_1",
            "task": "Search for customer complaints about battery life in reviews",
            "depends_on": []
        }},
        {{
            "id": "step_2",
            "task": "Search for battery-related support tickets",
            "depends_on": []
        }},
        {{
            "id": "step_3",
            "task": "Compare sentiment between review complaints and support tickets about batteries",
            "depends_on": ["step_1", "step_2"]
        }}
    ],
    "synthesis_instruction": "Combine the search results and sentiment comparison into a comprehensive analysis of battery-related customer issues across channels",
    "should_escalate": false,
    "escalation_reason": null
}}

Important:
- Each task should be a self-contained question a specialist agent can answer
- Use clear, specific language — the specialist agents have no context beyond the task
- For dependent steps, mention that prior results will be provided as context
- Set should_escalate=true for queries about: legal matters, PII, ambiguous requests,
  or anything where automated analysis might be insufficient"""

SYNTHESIS_PROMPT = """You are a Synthesis Agent combining results from multiple specialist agents
into one cohesive answer.

ORIGINAL USER QUERY: {query}

SYNTHESIS INSTRUCTION: {instruction}

SUB-TASK RESULTS:
{results}

Synthesize these results into a comprehensive, well-structured answer that directly
addresses the user's original query. Cite which sub-task provided each finding.
Be specific and data-driven — reference actual numbers, entities, and quotes from the results."""


# ── Escalation config ────────────────────────────────────────────────

ESCALATION_KEYWORDS = {
    "legal", "lawsuit", "compliance", "regulate", "regulation",
    "confidential", "pii", "personal data", "gdpr", "hipaa",
    "terminate", "fired", "layoff", "hr complaint",
}

# Critic score threshold below which we escalate
ESCALATION_SCORE_THRESHOLD = 2.5


class HierarchicalPlanner:
    """Orchestrates a hierarchy of specialist agents via plan-and-execute.

    Flow:
        User query → Planner (produces DAG) → DAG Executor (parallel sub-agents)
                   → Synthesizer (merges results) → Critic (optional) → Answer

    Supports human-in-the-loop escalation: when the planner or critic flags
    low confidence, the response is marked as escalated and the caller
    (API/frontend) can surface an approval prompt.
    """

    def __init__(
        self,
        api_key: str = None,
        documents_df: pd.DataFrame = None,
        model_name: str = DEFAULT_MODEL,
        max_sub_steps: int = 8,
        enable_critic: bool = True,
        enable_escalation: bool = True,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.documents_df = documents_df
        self.max_sub_steps = max_sub_steps
        self.enable_critic = enable_critic
        self.enable_escalation = enable_escalation

    def _check_escalation_keywords(self, query: str) -> Optional[str]:
        """Check if the query contains keywords that warrant escalation."""
        query_lower = query.lower()
        for keyword in ESCALATION_KEYWORDS:
            if keyword in query_lower:
                return f"Query contains sensitive topic: '{keyword}'"
        return None

    def _create_plan(self, query: str) -> ExecutionPlan:
        """Use Gemini to decompose the query into a DAG of sub-tasks."""
        # Check for escalation keywords first
        if self.enable_escalation:
            escalation_reason = self._check_escalation_keywords(query)
            if escalation_reason:
                return ExecutionPlan(
                    query=query,
                    steps=[PlanStep(id="step_1", task=query)],
                    synthesis_instruction="Direct answer",
                    should_escalate=True,
                    escalation_reason=escalation_reason,
                )

        prompt = PLANNER_PROMPT.format(query=query)

        try:
            response = generate_with_retry(
                self.client,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            parsed = json.loads(response.text)
            return self._parse_plan(query, parsed)

        except (json.JSONDecodeError, Exception):
            # Fallback: single-step plan (delegate everything to one agent)
            return ExecutionPlan(
                query=query,
                steps=[PlanStep(id="step_1", task=query)],
                synthesis_instruction="Provide a direct answer to the query",
            )

    def _parse_plan(self, query: str, parsed: dict) -> ExecutionPlan:
        """Convert parsed JSON into an ExecutionPlan."""
        steps = []
        for s in parsed.get("steps", []):
            steps.append(PlanStep(
                id=s.get("id", f"step_{len(steps)+1}"),
                task=s.get("task", ""),
                depends_on=s.get("depends_on", []),
            ))

        # Cap the number of steps
        steps = steps[:self.max_sub_steps]

        return ExecutionPlan(
            query=query,
            steps=steps,
            synthesis_instruction=parsed.get("synthesis_instruction", "Synthesize the results"),
            should_escalate=parsed.get("should_escalate", False),
            escalation_reason=parsed.get("escalation_reason", ""),
        )

    def _run_sub_agent(self, task: str, context: str = "") -> AgentResponse:
        """Run a specialist CustomerInsightAgent on a single sub-task."""
        agent = CustomerInsightAgent(
            api_key=self.api_key,
            documents_df=self.documents_df,
            model_name=self.model_name,
            max_steps=self.max_sub_steps,
            enable_critic=False,  # Critic runs at the planner level
        )

        full_query = task
        if context:
            full_query = f"{task}\n\nContext from prior analysis:\n{context}"

        return agent.query(full_query)

    async def _execute_dag(self, plan: ExecutionPlan) -> dict[str, str]:
        """Execute the plan DAG with parallel execution of independent steps.

        Uses topological ordering: steps with no unresolved dependencies
        run concurrently. As steps complete, their dependents become eligible.

        Returns:
            Dict mapping step_id → result string.
        """
        results: dict[str, str] = {}
        completed: set[str] = set()
        step_map = {s.id: s for s in plan.steps}

        while len(completed) < len(plan.steps):
            # Find steps whose dependencies are all satisfied
            ready = [
                s for s in plan.steps
                if s.id not in completed
                and s.status == StepStatus.PENDING
                and all(d in completed for d in s.depends_on)
            ]

            if not ready:
                # No steps are ready but not all complete — broken dependency
                break

            # Mark as running
            for s in ready:
                s.status = StepStatus.RUNNING

            # Execute ready steps in parallel
            async def _run_step(step: PlanStep) -> tuple[str, str]:
                # Build context from dependencies
                context_parts = []
                for dep_id in step.depends_on:
                    if dep_id in results:
                        dep_step = step_map[dep_id]
                        context_parts.append(f"[{dep_id}: {dep_step.task}]\n{results[dep_id]}")
                context = "\n\n".join(context_parts)

                response = await asyncio.to_thread(
                    self._run_sub_agent, step.task, context
                )
                step.agent_response = response
                return step.id, response.answer

            tasks = [_run_step(s) for s in ready]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in step_results:
                if isinstance(result, Exception):
                    # Find the failed step and mark it
                    for s in ready:
                        if s.status == StepStatus.RUNNING:
                            s.status = StepStatus.FAILED
                            s.result = f"Error: {result}"
                            results[s.id] = s.result
                            completed.add(s.id)
                            break
                else:
                    step_id, answer = result
                    step_map[step_id].status = StepStatus.COMPLETED
                    step_map[step_id].result = answer
                    results[step_id] = answer
                    completed.add(step_id)

        return results

    def _synthesize(self, query: str, instruction: str, results: dict[str, str],
                    plan: ExecutionPlan) -> str:
        """Synthesize sub-agent results into a final answer."""
        if len(results) == 1:
            # Single step — no need to synthesize
            return next(iter(results.values()))

        results_text = "\n\n".join(
            f"--- {step.id}: {step.task} ---\n{results.get(step.id, '(no result)')}"
            for step in plan.steps
        )

        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            instruction=instruction,
            results=results_text,
        )

        response = generate_with_retry(
            self.client,
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()

    async def aquery(self, query: str) -> PlannerResponse:
        """Async entry point: plan → execute DAG → synthesize → (critic) → answer."""

        # Step 1: Create the plan
        plan = self._create_plan(query)

        # Step 2: Check for escalation
        if plan.should_escalate and self.enable_escalation:
            # Still execute but mark as escalated
            results = await self._execute_dag(plan)
            answer = self._synthesize(query, plan.synthesis_instruction, results, plan)
            return PlannerResponse(
                answer=answer,
                plan=plan,
                sub_results=results,
                escalated=True,
                escalation_reason=plan.escalation_reason,
            )

        # Step 3: Execute the DAG
        results = await self._execute_dag(plan)

        # Step 4: Synthesize
        answer = self._synthesize(query, plan.synthesis_instruction, results, plan)

        # Step 5: Optional critic pass
        if self.enable_critic:
            try:
                critic = CriticAgent(api_key=self.api_key, model_name=self.model_name)
                evidence = [
                    (step.task, results.get(step.id, ""))
                    for step in plan.steps
                ]
                verdict = critic.evaluate(query=query, answer=answer, evidence=evidence)

                # Escalate on low scores
                if (self.enable_escalation
                        and verdict.overall_score < ESCALATION_SCORE_THRESHOLD):
                    return PlannerResponse(
                        answer=verdict.revised_answer or answer,
                        plan=plan,
                        sub_results=results,
                        escalated=True,
                        escalation_reason=f"Critic score {verdict.overall_score}/5 below threshold",
                    )

                if verdict.verdict != "PASS" and verdict.revised_answer:
                    answer = verdict.revised_answer
            except Exception:
                pass  # Critic failure should not block the answer

        return PlannerResponse(
            answer=answer,
            plan=plan,
            sub_results=results,
        )

    def query(self, query: str) -> PlannerResponse:
        """Synchronous entry point (runs the async pipeline in a new event loop)."""
        return asyncio.run(self.aquery(query))
