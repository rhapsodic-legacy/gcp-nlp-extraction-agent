"""Tests for the agent self-correction loop (critic feedback → revised answer)."""

from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pandas as pd
import pytest

from src.agent.agent import CustomerInsightAgent, AgentStep


@dataclass
class FakeVerdict:
    verdict: str = "PASS"
    overall_score: int = 5
    completeness_score: int = 5
    completeness_reason: str = ""
    grounding_score: int = 5
    grounding_reason: str = ""
    coherence_score: int = 5
    coherence_reason: str = ""
    revised_answer: str = ""


class TestFormatCritique:
    def test_pass_verdict(self):
        verdict = FakeVerdict(verdict="PASS", overall_score=5)
        result = CustomerInsightAgent._format_critique(verdict)
        assert "PASS" in result
        assert "5/5" in result

    def test_low_completeness_included(self):
        verdict = FakeVerdict(
            verdict="REVISE",
            overall_score=3,
            completeness_score=2,
            completeness_reason="Missing shipping data",
        )
        result = CustomerInsightAgent._format_critique(verdict)
        assert "Completeness" in result
        assert "Missing shipping data" in result

    def test_low_grounding_included(self):
        verdict = FakeVerdict(
            verdict="REVISE",
            overall_score=3,
            grounding_score=2,
            grounding_reason="No citations",
        )
        result = CustomerInsightAgent._format_critique(verdict)
        assert "Grounding" in result
        assert "No citations" in result

    def test_high_scores_omitted(self):
        verdict = FakeVerdict(
            verdict="PASS",
            overall_score=5,
            completeness_score=5,
            grounding_score=5,
            coherence_score=5,
        )
        result = CustomerInsightAgent._format_critique(verdict)
        # Only verdict line, no dimension details
        assert "Completeness" not in result
        assert "Grounding" not in result


class TestCriticEvaluate:
    @patch("src.agent.agent.genai")
    def _make_agent(self, mock_genai):
        df = pd.DataFrame([
            {"id": "1", "text": "test", "source_type": "review", "metadata": "{}"},
        ])
        return CustomerInsightAgent(api_key="fake", documents_df=df, enable_critic=True)

    @patch("src.agent.agent.generate_with_retry")
    @patch("src.agent.agent.genai")
    def test_pass_returns_original_answer(self, mock_genai, mock_gen):
        agent = self._make_agent()
        verdict = FakeVerdict(verdict="PASS")

        with patch("src.agent.critic.CriticAgent") as MockCritic:
            mock_critic_inst = MagicMock()
            mock_critic_inst.evaluate.return_value = verdict
            MockCritic.return_value = mock_critic_inst

            result = agent._critic_evaluate("query", "Original answer", [], [])

        assert result == "Original answer"

    @patch("src.agent.agent.generate_with_retry")
    @patch("src.agent.agent.genai")
    def test_revise_triggers_self_correction(self, mock_genai, mock_gen):
        agent = self._make_agent()
        verdict = FakeVerdict(
            verdict="REVISE",
            overall_score=3,
            completeness_score=2,
            completeness_reason="Incomplete",
        )

        mock_response = MagicMock()
        mock_response.text = "THOUGHT: Improving\nANSWER: Revised answer with more detail."
        mock_gen.return_value = mock_response

        with patch("src.agent.critic.CriticAgent") as MockCritic:
            mock_critic_inst = MagicMock()
            mock_critic_inst.evaluate.return_value = verdict
            MockCritic.return_value = mock_critic_inst

            result = agent._critic_evaluate("query", "Original", [], [])

        assert result == "Revised answer with more detail."

    @patch("src.agent.agent.generate_with_retry")
    @patch("src.agent.agent.genai")
    def test_critic_failure_returns_original(self, mock_genai, mock_gen):
        agent = self._make_agent()

        with patch("src.agent.critic.CriticAgent") as MockCritic:
            MockCritic.side_effect = Exception("Critic broke")

            result = agent._critic_evaluate("query", "Original answer", [], [])

        assert result == "Original answer"
        assert agent.last_critic_verdict is None

    @patch("src.agent.agent.generate_with_retry")
    @patch("src.agent.agent.genai")
    def test_fallback_to_critic_revised_answer(self, mock_genai, mock_gen):
        """If agent doesn't produce a revised answer, use critic's."""
        agent = self._make_agent()
        verdict = FakeVerdict(
            verdict="REVISE",
            overall_score=3,
            completeness_score=2,
            completeness_reason="Incomplete",
            revised_answer="Critic's better answer",
        )

        # Agent response doesn't contain ANSWER tag
        mock_response = MagicMock()
        mock_response.text = "THOUGHT: I'm not sure how to improve."
        mock_gen.return_value = mock_response

        with patch("src.agent.critic.CriticAgent") as MockCritic:
            mock_critic_inst = MagicMock()
            mock_critic_inst.evaluate.return_value = verdict
            MockCritic.return_value = mock_critic_inst

            result = agent._critic_evaluate("query", "Original", [], [])

        assert result == "Critic's better answer"
