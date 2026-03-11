"""Tests for the hierarchical planner, DAG execution, and escalation.

All Gemini calls are mocked — tests validate planning logic, DAG
topological execution, parallel batching, and escalation triggers.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

import pandas as pd
import pytest

from src.agent.planner import (
    HierarchicalPlanner,
    ExecutionPlan,
    PlanStep,
    PlannerResponse,
    StepStatus,
    ESCALATION_KEYWORDS,
    ESCALATION_SCORE_THRESHOLD,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_df():
    return pd.DataFrame([
        {"id": "1", "text": "Battery drains fast", "source_type": "review", "metadata": "{}"},
        {"id": "2", "text": "Shipping was late", "source_type": "support_ticket", "metadata": "{}"},
    ])


@patch("src.agent.planner.genai")
def _make_planner(mock_genai, **kwargs):
    return HierarchicalPlanner(
        api_key="fake",
        documents_df=_make_df(),
        **kwargs,
    )


# ── Plan creation tests ─────────────────────────────────────────────

class TestCreatePlan:
    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_creates_multi_step_plan(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "steps": [
                {"id": "step_1", "task": "Search reviews", "depends_on": []},
                {"id": "step_2", "task": "Search tickets", "depends_on": []},
                {"id": "step_3", "task": "Compare results", "depends_on": ["step_1", "step_2"]},
            ],
            "synthesis_instruction": "Combine findings",
            "should_escalate": False,
        })
        mock_gen.return_value = mock_response

        planner = _make_planner()
        plan = planner._create_plan("What are the top complaints?")

        assert len(plan.steps) == 3
        assert plan.steps[0].depends_on == []
        assert plan.steps[2].depends_on == ["step_1", "step_2"]
        assert plan.synthesis_instruction == "Combine findings"

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_fallback_single_step_on_error(self, mock_genai, mock_gen):
        mock_gen.side_effect = Exception("API error")

        planner = _make_planner()
        plan = planner._create_plan("Simple question")

        assert len(plan.steps) == 1
        assert plan.steps[0].task == "Simple question"

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_caps_max_steps(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "steps": [{"id": f"s{i}", "task": f"Task {i}", "depends_on": []} for i in range(20)],
            "synthesis_instruction": "Too many",
        })
        mock_gen.return_value = mock_response

        planner = _make_planner(max_sub_steps=5)
        plan = planner._create_plan("Complex query")

        assert len(plan.steps) <= 5


# ── Escalation tests ────────────────────────────────────────────────

class TestEscalation:
    def test_keyword_escalation(self):
        planner = _make_planner()
        for keyword in ["legal", "compliance", "GDPR", "pii"]:
            reason = planner._check_escalation_keywords(f"Query about {keyword} issues")
            assert reason is not None
            assert keyword.lower() in reason.lower()

    def test_no_escalation_for_normal_query(self):
        planner = _make_planner()
        reason = planner._check_escalation_keywords("What are the top product complaints?")
        assert reason is None

    @patch("src.agent.planner.genai")
    def test_escalation_keywords_trigger_plan_flag(self, mock_genai):
        planner = _make_planner(enable_escalation=True)
        plan = planner._create_plan("Are there any legal risks in these reviews?")

        assert plan.should_escalate is True
        assert "legal" in plan.escalation_reason.lower()

    @patch("src.agent.planner.genai")
    def test_escalation_disabled(self, mock_genai):
        planner = _make_planner(enable_escalation=False)
        reason = planner._check_escalation_keywords("legal question")
        # Method still works, but create_plan won't flag it
        assert reason is not None  # detection still works

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_critic_low_score_triggers_escalation(self, mock_genai, mock_gen):
        """Critic scores below threshold should escalate."""
        # Plan creation
        plan_response = MagicMock()
        plan_response.text = json.dumps({
            "steps": [{"id": "s1", "task": "Analyze data", "depends_on": []}],
            "synthesis_instruction": "Direct",
            "should_escalate": False,
        })

        # Sub-agent response
        agent_response = MagicMock()
        agent_response.text = "THOUGHT: done\nANSWER: Some answer."

        mock_gen.return_value = plan_response

        planner = _make_planner(enable_critic=True, enable_escalation=True)

        # Mock the sub-agent and critic
        with patch.object(planner, "_run_sub_agent") as mock_sub, \
             patch("src.agent.planner.CriticAgent") as MockCritic:
            mock_sub.return_value = MagicMock(answer="Sub-agent answer")

            mock_critic = MagicMock()
            mock_verdict = MagicMock()
            mock_verdict.verdict = "FAIL"
            mock_verdict.overall_score = 1.5  # below threshold
            mock_verdict.revised_answer = "Better answer"
            mock_critic.evaluate.return_value = mock_verdict
            MockCritic.return_value = mock_critic

            result = asyncio.run(planner.aquery("What's happening?"))

        assert result.escalated is True
        assert "score" in result.escalation_reason.lower()


# ── DAG execution tests ─────────────────────────────────────────────

class TestDAGExecution:
    def test_independent_steps_identified(self):
        """Steps with no dependencies should be identified as parallel-ready."""
        plan = ExecutionPlan(
            query="test",
            steps=[
                PlanStep(id="s1", task="Task 1", depends_on=[]),
                PlanStep(id="s2", task="Task 2", depends_on=[]),
                PlanStep(id="s3", task="Task 3", depends_on=["s1", "s2"]),
            ],
        )
        # Initially, s1 and s2 should be ready (no deps)
        ready = [s for s in plan.steps if not s.depends_on]
        assert len(ready) == 2
        assert {s.id for s in ready} == {"s1", "s2"}

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_dag_respects_dependencies(self, mock_genai, mock_gen):
        planner = _make_planner()

        call_order = []

        def mock_run_sub_agent(task, context=""):
            call_order.append(task)
            return MagicMock(answer=f"Result for: {task}")

        plan = ExecutionPlan(
            query="test",
            steps=[
                PlanStep(id="s1", task="First", depends_on=[]),
                PlanStep(id="s2", task="Second", depends_on=["s1"]),
            ],
        )

        with patch.object(planner, "_run_sub_agent", side_effect=mock_run_sub_agent):
            results = asyncio.run(planner._execute_dag(plan))

        assert "s1" in results
        assert "s2" in results
        # s1 must complete before s2 starts
        assert call_order.index("First") < call_order.index("Second")

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_dag_handles_step_failure(self, mock_genai, mock_gen):
        planner = _make_planner()

        def mock_run_sub_agent(task, context=""):
            if "fail" in task.lower():
                raise RuntimeError("Intentional failure")
            return MagicMock(answer=f"Result for: {task}")

        plan = ExecutionPlan(
            query="test",
            steps=[
                PlanStep(id="s1", task="This should fail", depends_on=[]),
                PlanStep(id="s2", task="This should succeed", depends_on=[]),
            ],
        )

        with patch.object(planner, "_run_sub_agent", side_effect=mock_run_sub_agent):
            results = asyncio.run(planner._execute_dag(plan))

        assert "s1" in results
        assert "Error" in results["s1"]
        assert "s2" in results
        assert "Result for" in results["s2"]

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_dependent_step_gets_context(self, mock_genai, mock_gen):
        """Steps with dependencies should receive prior results as context."""
        planner = _make_planner()
        received_contexts = []

        def mock_run_sub_agent(task, context=""):
            received_contexts.append(context)
            return MagicMock(answer=f"Result for: {task}")

        plan = ExecutionPlan(
            query="test",
            steps=[
                PlanStep(id="s1", task="Search for data", depends_on=[]),
                PlanStep(id="s2", task="Analyze the data", depends_on=["s1"]),
            ],
        )

        with patch.object(planner, "_run_sub_agent", side_effect=mock_run_sub_agent):
            asyncio.run(planner._execute_dag(plan))

        # s2 should have received s1's result as context
        assert len(received_contexts) == 2
        assert received_contexts[0] == ""  # s1 has no deps
        assert "Result for: Search for data" in received_contexts[1]  # s2 gets s1's result


# ── Synthesis tests ──────────────────────────────────────────────────

class TestSynthesis:
    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_single_result_skips_synthesis(self, mock_genai, mock_gen):
        planner = _make_planner()
        plan = ExecutionPlan(
            query="test",
            steps=[PlanStep(id="s1", task="Only task")],
        )
        result = planner._synthesize("test", "Direct", {"s1": "The answer"}, plan)
        assert result == "The answer"
        # No LLM call needed for single result
        mock_gen.assert_not_called()

    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_multi_result_calls_synthesizer(self, mock_genai, mock_gen):
        mock_response = MagicMock()
        mock_response.text = "Synthesized answer combining both results."
        mock_gen.return_value = mock_response

        planner = _make_planner()
        plan = ExecutionPlan(
            query="test",
            steps=[
                PlanStep(id="s1", task="Task 1"),
                PlanStep(id="s2", task="Task 2"),
            ],
        )
        result = planner._synthesize(
            "test", "Combine", {"s1": "Result 1", "s2": "Result 2"}, plan
        )
        assert result == "Synthesized answer combining both results."
        mock_gen.assert_called_once()


# ── End-to-end query test ────────────────────────────────────────────

class TestPlannerQuery:
    @patch("src.agent.planner.generate_with_retry")
    @patch("src.agent.planner.genai")
    def test_full_pipeline(self, mock_genai, mock_gen):
        # Plan creation response
        plan_json = json.dumps({
            "steps": [
                {"id": "s1", "task": "Find complaints", "depends_on": []},
            ],
            "synthesis_instruction": "Direct",
            "should_escalate": False,
        })

        call_count = [0]
        def mock_generate(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.text = plan_json  # planning call
            else:
                resp.text = "THOUGHT: done\nANSWER: Found 3 complaints about batteries."
            return resp

        mock_gen.side_effect = mock_generate

        planner = _make_planner(enable_critic=False)
        result = planner.query("What are the complaints?")

        assert isinstance(result, PlannerResponse)
        assert result.answer is not None
        assert len(result.plan.steps) == 1
        assert not result.escalated
